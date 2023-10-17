import pandas as pd


def process_features(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    processed = process_sex(processed)
    processed = process_name(processed)
    processed = process_age(processed)
    processed = process_animal(processed)

    return processed


def process_name(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    processed["Is_Nan_name"] = features["Name"].isna().astype(int).astype("category")

    return processed


def process_sex(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    splitted_sex = processed["SexuponOutcome"].str.split(expand=True)
    splitted_sex.columns = ["Condition", "Sex"]
    splitted_sex.fillna("Unknown", inplace=True)

    splitted_sex["Condition"] = splitted_sex["Condition"].replace("Spayed", "Neutered")

    splitted_sex["Condition"] = splitted_sex["Condition"].astype("category")
    splitted_sex["Sex"] = splitted_sex["Sex"].astype("category")

    processed = pd.concat([processed, splitted_sex], axis=1)
    processed.drop(columns="SexuponOutcome", inplace=True)

    return processed


def process_animal(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    processed["AnimalType"] = processed["AnimalType"].astype("category")

    return processed


def process_age(features: pd.DataFrame) -> pd.DataFrame:
    processed = features.copy()

    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("years", "year")
    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("months", "month")
    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("weeks", "week")
    processed["AgeuponOutcome"] = processed["AgeuponOutcome"].str.replace("days", "day")

    splitted_age = processed["AgeuponOutcome"].str.split(expand=True)
    splitted_age.columns = ["Num_age", "Magnitude_age"]

    splitted_age["Num_age"] = splitted_age["Num_age"].fillna(-1).astype("int")

    is_zero = splitted_age["Num_age"] == 0
    splitted_age.loc[is_zero, "Num_age"] = -1

    splitted_age.loc[splitted_age["Magnitude_age"] == "week", "Num_age"] = (
        splitted_age.loc[splitted_age["Magnitude_age"] == "week", "Num_age"] * 7
    )
    splitted_age.loc[splitted_age["Magnitude_age"] == "month", "Num_age"] = (
        splitted_age.loc[splitted_age["Magnitude_age"] == "month", "Num_age"] * 30
    )
    splitted_age.loc[splitted_age["Magnitude_age"] == "year", "Num_age"] = (
        splitted_age.loc[splitted_age["Magnitude_age"] == "year", "Num_age"] * 365
    )

    splitted_age["Magnitude_age"] = splitted_age["Magnitude_age"].fillna("Unknown")

    splitted_age["Magnitude_age"] = splitted_age["Magnitude_age"].astype("category")
    splitted_age["Num_age"] = splitted_age["Num_age"].astype("int")

    processed = pd.concat([processed, splitted_age], axis=1)
    processed.drop(columns="AgeuponOutcome", inplace=True)

    return processed


def process_time(features: pd.DataFrame, time: pd.Series) -> pd.DataFrame:
    # We can do this because test split is not timeseries split

    time = time.copy()
    processed = features.copy()

    time_index = pd.to_datetime(time)
    processed["year"] = time_index.dt.year
    processed["month"] = time_index.dt.month
    processed["dow"] = time_index.dt.dayofweek
    processed["hour"] = time_index.dt.hour

    return processed
