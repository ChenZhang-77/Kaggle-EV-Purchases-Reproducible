"""Train the two selected entries from the four source CSV files."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from features import (
    TARGET, BaselineFeatures, catboost_categories, encoded_frames,
    prepare_public_features,
)

SEEDS = (42, 2026)
FOLDS = 5
FEATURE_NAMES = (
    "lgb_seed42", "lgb_seed2026", "round8",
    "grouped_constraints", "catboost_native_public",
)
LGB_PARAMS = dict(
    n_estimators=20000, learning_rate=0.02, max_depth=5, num_leaves=32,
    min_child_samples=10, subsample=0.812763, colsample_bytree=0.30293,
    reg_alpha=0.07094, reg_lambda=2.03303, max_bin=1024,
    feature_pre_filter=False, metric="auc", n_jobs=4, verbosity=-1,
)
XGB_COMMON = dict(
    subsample=0.85, colsample_bytree=0.9, tree_method="hist", max_bin=512,
)
XGB_SMOOTH = dict(
    n_estimators=500, learning_rate=0.06, max_depth=6,
    min_child_weight=10, reg_lambda=5, **XGB_COMMON,
)
XGB_DEPTH5 = dict(
    n_estimators=700, learning_rate=0.05, max_depth=5,
    min_child_weight=10, reg_lambda=5, **XGB_COMMON,
)
CAT_BASELINE = dict(
    iterations=800, depth=6, learning_rate=0.06,
    l2_leaf_reg=5, border_count=254,
)
CAT_NATIVE = dict(
    iterations=4000, depth=7, learning_rate=0.04, l2_leaf_reg=8.0,
    random_strength=0.5, border_count=128, bootstrap_type="Bayesian",
    bagging_temperature=0.5, loss_function="Logloss", eval_metric="AUC",
    od_type="Iter", od_wait=300,
)
SOURCE_COLUMNS = (
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level", "Annual_Income_USD", "Daily_Commute_km",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
    "Current_Car_Type", "City_Type", "Gender", "Age",
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def rank(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rank input must be finite and one-dimensional")
    return rankdata(values, method="average") / len(values)


def fold_rank(values, folds):
    result = np.empty(len(values), dtype=float)
    for fold in np.unique(folds):
        selected = folds == fold
        result[selected] = rank(values[selected])
    return result


def fold_ids(y, seed):
    result = np.full(len(y), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y)):
        result[valid] = fold
    return result


def source_for_feature(name):
    for source in SOURCE_COLUMNS:
        if name.startswith(source):
            return source
    if name.startswith(("income", "is_30k", "is_millionaire", "is_dead_zone")):
        return "Annual_Income_USD"
    if name.startswith("commute_integer"):
        return "Daily_Commute_km"
    if name.startswith("is_env_hater"):
        return "Environmental_Concern_Level"
    raise ValueError(f"Unknown feature group for {name}")


def interaction_groups(columns):
    groups = {source: [] for source in SOURCE_COLUMNS}
    for index, name in enumerate(columns):
        groups[source_for_feature(name)].append(index)
    result = [group for group in groups.values() if group]
    assert sorted(index for group in result for index in group) == list(range(len(columns)))
    return result


def xgb_model(params, seed, threads, train_frame):
    numeric = list(train_frame.select_dtypes(include="number"))
    categorical = [name for name in train_frame if name not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", XGBClassifier(
            **params, random_state=seed, n_jobs=threads,
            objective="binary:logistic", eval_metric="logloss",
        )),
    ])


def baseline_fit(train, y, valid, test, seed, member, threads):
    smooth = member == "smooth"
    builder = BaselineFeatures(
        seed=seed, encodings=["income"] if smooth else ["income", "income_bin500"],
        smooth=5 if smooth else 20,
    )
    A = builder.fit_transform(train, y)
    B = builder.transform(valid)
    C = builder.transform(test)
    if member == "catboost":
        A, B, C = map(catboost_categories, (A, B, C))
        categorical = list(A.select_dtypes(exclude="number"))
        model = CatBoostClassifier(
            **CAT_BASELINE, cat_features=categorical, loss_function="Logloss",
            random_seed=seed, thread_count=threads, verbose=False,
            allow_writing_files=False,
        )
    else:
        model = xgb_model(XGB_SMOOTH if smooth else XGB_DEPTH5, seed, threads, A)
    with threadpool_limits(limits=threads):
        model.fit(A, y)
        return model.predict_proba(B)[:, 1], model.predict_proba(C)[:, 1]


def public_fit(X, y, V, y_valid, T, columns, seed, threads, grouped):
    A, B, C = encoded_frames(X, y, V, T, columns, seed)
    params = dict(LGB_PARAMS, random_state=seed, n_jobs=threads)
    if grouped:
        params["interaction_constraints"] = interaction_groups(A.columns)
    model = lgb.LGBMClassifier(**params)
    with threadpool_limits(limits=threads):
        model.fit(A, y, eval_set=[(B, y_valid)],
                  callbacks=[lgb.early_stopping(500, verbose=False)])
        return model.predict_proba(B)[:, 1], model.predict_proba(C)[:, 1]


def native_cat_fit(X, y, V, y_valid, T, seed, device, threads):
    A, B, C = map(catboost_categories, (X, V, T))
    categorical = list(A.select_dtypes(exclude="number"))
    params = dict(CAT_NATIVE, task_type=device)
    if device == "GPU":
        params["devices"] = "0"
    model = CatBoostClassifier(
        **params, cat_features=categorical, random_seed=seed,
        thread_count=threads, verbose=False, allow_writing_files=False,
    )
    model.fit(A, y, eval_set=(B, y_valid), use_best_model=True)
    return model.predict_proba(B)[:, 1], model.predict_proba(C)[:, 1]


def cached_fold(cache, valid, test_rows, trainer):
    if cache.exists():
        with np.load(cache) as saved:
            np.testing.assert_array_equal(saved["valid"], valid)
            probability = saved["valid_probability"]
            test_probability = saved["test_probability"]
        if probability.shape != (len(valid),) or test_probability.shape != (test_rows,):
            raise ValueError(f"Cached prediction shape mismatch: {cache}")
    else:
        probability, test_probability = trainer()
        if not np.isfinite(probability).all() or not np.isfinite(test_probability).all():
            raise ValueError(f"Nonfinite prediction: {cache}")
        np.savez_compressed(
            cache, valid=valid, valid_probability=probability,
            test_probability=test_probability,
        )
    return probability, test_probability


def run_cv(name, seed, folds, train, test, y, output, trainer):
    directory = output / "cache" / name / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    oof = np.empty(len(train))
    test_values = np.zeros(len(test))
    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        cache = directory / f"fold_{fold}.npz"
        probability, test_probability = cached_fold(
            cache, valid, len(test),
            lambda: trainer(fit, valid),
        )
        oof[valid] = probability
        test_values += test_probability / FOLDS
        print(f"{name} seed={seed} fold={fold} AUC={roc_auc_score(y[valid], probability):.6f}", flush=True)
    return oof, test_values


def fit_nonnegative_logistic(X, y, l2=1e-6):
    prior = np.clip(y.mean(), 1e-8, 1 - 1e-8)
    initial = np.r_[np.log(prior / (1 - prior)), np.ones(X.shape[1])]

    def objective(parameters):
        score = parameters[0] + X @ parameters[1:]
        loss = np.mean(np.logaddexp(0.0, score) - y * score)
        loss += 0.5 * l2 * np.dot(parameters[1:], parameters[1:])
        residual = expit(score) - y
        gradient = np.r_[
            np.mean(residual), X.T @ residual / len(y) + l2 * parameters[1:]
        ]
        return loss, gradient

    result = minimize(
        objective, initial, method="L-BFGS-B", jac=True,
        bounds=[(None, None)] + [(0.0, None)] * X.shape[1],
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[0], result.x[1:]


def validate_submission(frame, sample):
    if list(frame) != list(sample) or not frame.id.equals(sample.id):
        raise ValueError("Submission IDs or columns differ from sample")
    values = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Submission contains invalid probabilities")


def save_submission(sample, values, path):
    result = sample.copy()
    result[TARGET] = values
    validate_submission(result, sample)
    result.to_csv(path, index=False)
    return digest(path)


def load_inputs(data_dir):
    paths = {name: data_dir / name for name in
             ("train.csv", "test.csv", "sample_submission.csv", "original.csv")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    train, test, sample, original = (pd.read_csv(paths[name]) for name in paths)
    if list(test) != [c for c in train if c != TARGET]:
        raise ValueError("Train and test columns differ")
    if list(sample) != ["id", TARGET] or not sample.id.equals(test.id):
        raise ValueError("Sample IDs differ from test")
    for frame in (train, original):
        if TARGET not in frame or not frame[TARGET].isin(["Yes", "No"]).all():
            raise ValueError("Expected Yes/No target in train and original")
    return paths, train, test, sample, original


def run(args):
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    paths, train, test, sample, original = load_inputs(data_dir)
    if output == data_dir or data_dir in output.parents:
        raise ValueError("Output directory must be separate from the input files")
    manifest = {
        "inputs": {name: digest(path) for name, path in paths.items()},
        "source": {name: digest(Path(__file__).with_name(name)) for name in
                   ("features.py", "reproduce.py")},
        "device": args.catboost_device,
        "threads": args.threads,
        "model_params": {
            "lightgbm": LGB_PARAMS,
            "xgboost_smooth": XGB_SMOOTH,
            "xgboost_depth5": XGB_DEPTH5,
            "catboost_baseline": CAT_BASELINE,
            "catboost_native": CAT_NATIVE,
        },
        "packages": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost")},
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty and has no manifest")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Inputs, code or environment changed; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    y = train[TARGET].eq("Yes").to_numpy(dtype=int)
    fold_sets = {seed: fold_ids(y, seed) for seed in SEEDS}
    base_columns = [c for c in test if c != "id"]
    baseline_oof = {}
    baseline_test = {}
    for seed in SEEDS:
        member_oof, member_test = {}, {}
        for member in ("smooth", "depth5", "catboost"):
            member_oof[member], member_test[member] = run_cv(
                f"baseline_{member}", seed, fold_sets[seed], train, test, y, output,
                lambda fit, valid, member=member, seed=seed: baseline_fit(
                    train.iloc[fit][base_columns], y[fit],
                    train.iloc[valid][base_columns], test[base_columns],
                    seed, member, args.threads,
                ),
            )
        baseline_oof[seed] = (
            2 * fold_rank((member_oof["smooth"] + member_oof["depth5"]) / 2,
                          fold_sets[seed])
            + fold_rank(member_oof["catboost"], fold_sets[seed])
        ) / 3
        if seed == 42:
            # The original selected baseline used full-data models for test predictions.
            full = {}
            for member in ("smooth", "depth5", "catboost"):
                cache = output / "cache" / f"baseline_full_{member}.npy"
                if cache.exists():
                    full[member] = np.load(cache)
                else:
                    full[member] = baseline_fit(
                        train[base_columns], y, test[base_columns], test[base_columns],
                        seed, member, args.threads,
                    )[1]
                    np.save(cache, full[member])
            baseline_test[42] = (
                2 * rank((full["smooth"] + full["depth5"]) / 2)
                + rank(full["catboost"])
            ) / 3

    public_train, public_test, features, target_columns = prepare_public_features(
        train, test, original,
    )
    X, T = public_train[features], public_test[features]
    lgb_oof, lgb_test = {}, {}
    for seed in SEEDS:
        lgb_oof[seed], lgb_test[seed] = run_cv(
            "public_lgb", seed, fold_sets[seed], public_train, public_test, y, output,
            lambda fit, valid, seed=seed: public_fit(
                X.iloc[fit], y[fit], X.iloc[valid], y[valid],
                T, target_columns, seed, args.threads, False,
            ),
        )

    averaged_lgb = (rank(lgb_test[42]) + rank(lgb_test[2026])) / 2
    primary = .75 * averaged_lgb + .25 * rank(baseline_test[42])
    primary_sha = save_submission(sample, primary, output / "submission_weighted.csv")

    grouped_oof, grouped_test = run_cv(
        "grouped_lgb", 42, fold_sets[42], public_train, public_test, y, output,
        lambda fit, valid: public_fit(
            X.iloc[fit], y[fit], X.iloc[valid], y[valid],
            T, target_columns, 42, args.threads, True,
        ),
    )
    native_oof, native_test = run_cv(
        "native_catboost", 42, fold_sets[42], public_train, public_test, y, output,
        lambda fit, valid: native_cat_fit(
            X.iloc[fit], y[fit], X.iloc[valid], y[valid],
            T, 42, args.catboost_device, args.threads,
        ),
    )
    oof_columns = [lgb_oof[42], lgb_oof[2026], baseline_oof[42], grouped_oof, native_oof]
    test_columns = [lgb_test[42], lgb_test[2026], baseline_test[42], grouped_test, native_test]
    # The second seed used a distinct Round 8 OOF partition in the historical stack.
    stacking_scores = []
    for seed in SEEDS:
        columns = oof_columns.copy()
        columns[2] = baseline_oof[seed]
        X_stack = np.column_stack([rank(column) - .5 for column in columns])
        T_stack = np.column_stack([rank(column) - .5 for column in test_columns])
        score = np.zeros(len(test))
        for fold in range(FOLDS):
            fit = fold_sets[seed] != fold
            intercept, weights = fit_nonnegative_logistic(X_stack[fit], y[fit])
            score += (intercept + T_stack @ weights) / FOLDS
        stacking_scores.append(rank(score))
    stacked = np.mean(stacking_scores, axis=0)
    stacking_sha = save_submission(sample, stacked, output / "submission_stacking.csv")
    summary = {
        "weighted_sha256": primary_sha,
        "stacking_sha256": stacking_sha,
        "weighted_oof_auc_by_seed": {
            str(seed): float(roc_auc_score(
                y, .75 * (rank(lgb_oof[42]) + rank(lgb_oof[2026])) / 2
                + .25 * rank(baseline_oof[seed])
            )) for seed in SEEDS
        },
        "feature_names": list(FEATURE_NAMES),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/final"))
    parser.add_argument("--catboost-device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
