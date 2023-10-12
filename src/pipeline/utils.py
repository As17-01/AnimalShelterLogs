import numpy as np
import pandas as pd


def process_features(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    processed = process_sex(processed)
    processed = process_age(processed)

    return processed


def process_sex(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    splitted_sex = processed["SexuponOutcome"].str.split(expand=True)
    splitted_sex.columns = ["Condition", "Sex"]
    splitted_sex.fillna("Unknown", inplace=True)

    processed = pd.concat([processed, splitted_sex], axis=1)
    processed.drop(columns="SexuponOutcome", inplace=True)

    return processed


def process_age(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("years", "year")
    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("months", "month")
    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("weeks", "week")
    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("days", "day")

    splitted_age = processed["AgeuponOutcome"].str.split(expand=True)
    splitted_age.columns = ["num_age", "magnitude_age"]

    splitted_age["num_age"] = splitted_age["num_age"].astype("float")

    is_year = splitted_age["magnitude_age"] == "year"
    splitted_age.loc[is_year, "num_age"] = 365 * splitted_age.loc[is_year, "num_age"]
    is_month = splitted_age["magnitude_age"] == "month"
    splitted_age.loc[is_month, "num_age"] = 30 * splitted_age.loc[is_month, "num_age"]
    is_week = splitted_age["magnitude_age"] == "week"
    splitted_age.loc[is_week, "num_age"] = 7 * splitted_age.loc[is_week, "num_age"]

    mean_num_age = splitted_age["num_age"].mean(skipna=True)

    splitted_age["num_age"] = splitted_age["num_age"].fillna(mean_num_age)
    is_zero = splitted_age["num_age"] == 0.0
    splitted_age.loc[is_zero, "num_age"] = mean_num_age

    splitted_age["num_age"] = np.log(splitted_age["num_age"])
    splitted_age["num_age"] = splitted_age["num_age"] - np.mean(splitted_age["num_age"])

    splitted_age["magnitude_age"] = splitted_age["magnitude_age"].fillna("Unknown")

    processed = pd.concat([processed, splitted_age], axis=1)
    processed.drop(columns="AgeuponOutcome", inplace=True)

    return processed
