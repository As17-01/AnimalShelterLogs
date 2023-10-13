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
    splitted_age.columns = ["Num_age", "Magnitude_age"]

    splitted_age["Num_age"] = splitted_age["Num_age"].astype("float")

    is_year = splitted_age["Magnitude_age"] == "year"
    splitted_age.loc[is_year, "Num_age"] = 365 * splitted_age.loc[is_year, "Num_age"]
    is_month = splitted_age["Magnitude_age"] == "month"
    splitted_age.loc[is_month, "Num_age"] = 30 * splitted_age.loc[is_month, "Num_age"]
    is_week = splitted_age["Magnitude_age"] == "week"
    splitted_age.loc[is_week, "Num_age"] = 7 * splitted_age.loc[is_week, "Num_age"]

    mean_num_age = splitted_age["Num_age"].mean(skipna=True)

    splitted_age["Num_age"] = splitted_age["Num_age"].fillna(mean_num_age)
    is_zero = splitted_age["Num_age"] == 0.0
    splitted_age.loc[is_zero, "Num_age"] = mean_num_age

    splitted_age["Num_age"] = np.log(splitted_age["Num_age"])
    splitted_age["Num_age"] = splitted_age["Num_age"] - np.mean(splitted_age["Num_age"])

    splitted_age["Magnitude_age"] = splitted_age["Magnitude_age"].fillna("Unknown")

    processed = pd.concat([processed, splitted_age], axis=1)
    processed.drop(columns="AgeuponOutcome", inplace=True)

    return processed
