from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd


class CategoricalEncoder:
    def __init__(self, col_name: str, callback: Callable, rare_threshold: int = 20) -> None:
        self.col_name = col_name
        self.callback = callback
        self.mapping: Dict[str, int] = {}
        self.rare_threshold = rare_threshold

    def fit(self, features: pd.DataFrame):
        features, cat_cols = self.callback(features)

        melted = features[cat_cols].melt(value_name="value")
        mapping = melted.groupby("value")["value"].count()

        self.mapping = mapping[mapping > self.rare_threshold].to_dict()

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features, cat_cols = self.callback(features)

        for col_name in cat_cols:
            features[col_name] = features[col_name].map(lambda x: self.mapping[x] if x in self.mapping else 1)
            features[col_name] = features[col_name].astype("category")
        return features


def process_names(features) -> Tuple[pd.DataFrame, List[str]]:
    features = features.copy()

    features["Name"] = features["Name"].str.replace(" ", "")
    return features, ["Name"]


def process_colors(features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    features = features.copy()

    num_colors = (features["Color"].str.count("/") + 1).astype("int")
    first_color = features["Color"].str.split(pat="/", expand=True)[0].fillna("No")
    second_color = features["Color"].str.split(pat="/", expand=True)[1].fillna("No")
    is_complex = (first_color.str.count(" ") + second_color.str.count(" ")).astype("int")

    colors = pd.concat([first_color, second_color, num_colors, is_complex], axis=1)
    colors.columns = ["First_color", "Second_color", "Num_colors", "Is_complex_color"]

    features = pd.concat([features, colors], axis=1)
    features.drop(columns="Color", inplace=True)

    return features, ["First_color", "Second_color"]


def process_breeds(features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    features = features.copy()

    num_breeds = (features["Breed"].str.count("/") + 1).astype("int")
    first_breed = features["Breed"].str.split(pat="/", expand=True)[0].fillna("No")
    # second_breed = features["Breed"].str.split(pat="/", expand=True)[1].fillna("No")
    is_complex = (first_breed.str.count(" ")).astype("int")

    breeds = pd.concat([first_breed, num_breeds, is_complex], axis=1)
    breeds.columns = ["First_breed", "Num_breeds", "Is_complex_breed"]

    features = pd.concat([features, breeds], axis=1)
    features.drop(columns="Breed", inplace=True)

    return features, ["First_breed"]
