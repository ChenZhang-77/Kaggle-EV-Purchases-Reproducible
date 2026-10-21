"""Feature transforms used by the selected EV purchase submissions.

The public LightGBM feature recipe below adapts Naji's Kaggle notebook
(linked in README.md). This implementation changes its structure to support
cross-fitted training, configurable data paths, and both selected ensembles.
"""

import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

TARGET = "Will_Buy_EV"
SMOOTHING = ("auto", 10.0, 100.0)


def baseline_static(frame):
    result = frame.copy()
    result["Income_Last_Two_Digits"] = frame.Annual_Income_USD.round().mod(100)
    result["Commute_Tenth_Digit"] = (frame.Daily_Commute_km * 10).round().mod(10)
    result["Income_Minimum_Flag"] = frame.Annual_Income_USD.eq(30000).astype(int)
    result["Commute_Minimum_Flag"] = frame.Daily_Commute_km.eq(5).astype(int)
    result["Income_Last_Three_Digits"] = frame.Annual_Income_USD.round().mod(1000)
    result["Commute_Units_Digit"] = frame.Daily_Commute_km.round().mod(10)
    return result


class BaselineFeatures:
    """The selected XGBoost/CatBoost fold features, including cross-fitted TE."""

    def __init__(self, seed, encodings, smooth):
        self.seed = seed
        self.encodings = encodings
        self.smooth = smooth

    def keys(self, frame):
        income = frame.Annual_Income_USD.round().astype("int64")
        result = pd.DataFrame(index=frame.index)
        if "income" in self.encodings:
            result["Income_Target_Mean"] = income
        if "income_bin500" in self.encodings:
            result["Income_Bin500_Target_Mean"] = income // 500
        return result

    def fit_transform(self, frame, y):
        self.counts = frame.Annual_Income_USD.value_counts(dropna=False)
        self.size = len(frame)
        self.encoder = TargetEncoder(
            target_type="binary", smooth=self.smooth,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
        )
        values = self.encoder.fit_transform(self.keys(frame), y)
        return self._combine(frame, values)

    def transform(self, frame):
        return self._combine(frame, self.encoder.transform(self.keys(frame)))

    def _combine(self, frame, values):
        result = baseline_static(frame)
        result["Annual_Income_USD_Train_Frequency"] = (
            frame.Annual_Income_USD.map(self.counts).fillna(0).astype(float) / self.size
        )
        for index, name in enumerate(self.keys(frame)):
            result[name] = values[:, index]
        return result


def catboost_categories(frame):
    result = frame.copy()
    for name in result.select_dtypes(exclude="number"):
        result[name] = result[name].fillna("__MISSING__").astype(str)
    return result


def prepare_public_features(train, test, original):
    """Adapted from the cited public LightGBM notebook and project experiments."""
    train, test, original = train.copy(), test.copy(), original.copy()
    train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
    original[TARGET] = original[TARGET].map({"Yes": 1, "No": 0})
    train["is_train"], test["is_train"] = 1, 0
    test[TARGET] = np.nan
    combined = pd.concat([train, test], ignore_index=True)
    combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")
    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in combined if c not in categorical + ["id", "is_train", TARGET]]

    digits = {}
    for column in list(numeric):
        for position in range(-4, 4):
            digits[f"{column}_digit{position}"] = (
                combined[column].fillna(0) // (10 ** position) % 10
            ).astype("int8")
    combined = pd.concat([combined, pd.DataFrame(digits, index=combined.index)], axis=1)
    numeric.extend(digits)

    prior = float(original[TARGET].mean())
    original_means = {}
    for column in categorical + numeric:
        if column in original:
            mapping = original.groupby(column, observed=False)[TARGET].mean()
            original_means[f"{column}_org_mean"] = (
                combined[column].map(mapping).fillna(prior).astype("float32")
            )
    combined = pd.concat([combined, pd.DataFrame(original_means, index=combined.index)], axis=1)

    category_data = {
        f"{column}_cat": combined[column].fillna("NaN").astype(str)
        for column in numeric
    }
    combined = pd.concat([combined, pd.DataFrame(category_data, index=combined.index)], axis=1)
    target_columns = categorical + list(category_data)
    extras = {}
    for column in target_columns:
        extras[f"{column}_fe"] = (
            combined[column].map(combined[column].value_counts(normalize=True))
            .fillna(0).astype("float32")
        )
    income, commute = combined.Annual_Income_USD, combined.Daily_Commute_km
    extras.update({
        "is_30k_spike": income.eq(30000).astype("int8"),
        "is_millionaire_cliff": income.ge(170537).astype("int8"),
        "is_dead_zone": income.between(38000, 42000).astype("int8"),
        "is_env_hater": combined.Environmental_Concern_Level.eq(1).astype("int8"),
        "income_exact_int": np.floor(income).astype(str),
        "income100_floor": np.floor(income / 100).astype(str),
        "income1000_floor": np.floor(income / 1000).astype(str),
        "commute_integer": np.floor(commute).astype(str),
    })
    combined = pd.concat([combined, pd.DataFrame(extras, index=combined.index)], axis=1)
    target_columns.extend(["income_exact_int", "income100_floor", "income1000_floor", "commute_integer"])

    result_train = combined[combined.is_train.eq(1)].drop(columns="is_train").reset_index(drop=True)
    result_test = combined[combined.is_train.eq(0)].drop(columns=["is_train", TARGET]).reset_index(drop=True)
    numeric_columns = [
        c for c in result_train if c not in ["id", TARGET]
        and pd.api.types.is_numeric_dtype(result_train[c])
    ]
    correlation = result_train[numeric_columns].corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    correlated = [c for c in upper if (upper[c] == 1).any()]
    constant = [c for c in result_train if result_train[c].nunique() == 1]
    constant += [c for c in result_test if result_test[c].nunique() == 1]
    dropped = set(correlated + constant) - {"id", TARGET}
    result_train.drop(columns=dropped, inplace=True, errors="ignore")
    result_test.drop(columns=dropped, inplace=True, errors="ignore")
    target_columns = [c for c in target_columns if c not in dropped]
    features = [c for c in result_test if c != "id"]
    return result_train, result_test, features, target_columns


def encoded_frames(X_train, y_train, X_valid, X_test, target_columns, seed):
    train_parts = [X_train.drop(columns=target_columns)]
    valid_parts = [X_valid.drop(columns=target_columns)]
    test_parts = [X_test.drop(columns=target_columns)]
    for smooth in SMOOTHING:
        encoder = TargetEncoder(
            target_type="binary", shuffle=True, cv=5, smooth=smooth,
            random_state=seed,
        )
        a = encoder.fit_transform(X_train[target_columns], y_train).astype("float32")
        b = encoder.transform(X_valid[target_columns]).astype("float32")
        c = encoder.transform(X_test[target_columns]).astype("float32")
        suffix = "auto" if smooth == "auto" else str(int(smooth))
        names = [f"{column}_TE_{suffix}" for column in target_columns]
        train_parts.append(pd.DataFrame(a, columns=names, index=X_train.index))
        valid_parts.append(pd.DataFrame(b, columns=names, index=X_valid.index))
        test_parts.append(pd.DataFrame(c, columns=names, index=X_test.index))
        del encoder, a, b, c
        gc.collect()
    return tuple(pd.concat(parts, axis=1).astype("float32") for parts in
                 (train_parts, valid_parts, test_parts))
