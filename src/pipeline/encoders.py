from typing import Dict

import numpy as np
import pandas as pd


RARE_THRESHOLD = 50

class NameEncoder:
    def __init__(self):
        self.mapping: Dict[str, int] = {}

    def fit(self, features: pd.DataFrame):
        features = features.copy()

        features["Name"] = features["Name"].str.replace(" ", "")

        mapping = features.groupby("Name")["Name"].count()
        self.mapping = mapping[mapping > RARE_THRESHOLD].to_dict()

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.copy()

        cur_mapping = self.mapping.copy()
        for name in features["Name"].unique():
            if name not in cur_mapping:
                cur_mapping[name] = 1

        features = features.replace({"Name": cur_mapping})
        return features


class ColorEncoder:
    def __init__(self):
        self.mapping: Dict[str, int] = {}

    @staticmethod
    def process_colors(features: pd.DataFrame) -> pd.DataFrame:
        num_colors = features["Color"].str.count("/") + 1

        colors = features["Color"].str.split(pat="/", expand=True)[0]
        colors = pd.concat([colors, num_colors], axis=1)
        colors.columns = ["First_color", "Num_colors"]

        colors["Is_complex_color"] = colors["First_color"].str.count(" ")

        # Just in case
        colors["First_color"] = colors["First_color"].fillna("No")

        return colors

    def fit(self, features: pd.DataFrame):
        features = features.copy()
        colors = self.process_colors(features)

        mapping = colors.groupby("First_color")["First_color"].count()
        self.mapping = mapping[mapping > RARE_THRESHOLD].to_dict()

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.copy()
        colors = self.process_colors(features)

        cur_mapping = self.mapping.copy()
        for clr in colors["First_color"].unique():
            if clr not in cur_mapping:
                cur_mapping[clr] = 1

        colors = colors.replace({"First_color": cur_mapping})

        features = pd.concat([features, colors], axis=1)
        features.drop(columns="Color", inplace=True)

        return features


class BreedEncoder:
    def __init__(self):
        self.mapping: Dict[str, int] = {}

    @staticmethod
    def process_breeds(features: pd.DataFrame) -> pd.DataFrame:
        num_breeds = features["Breed"].str.count("/") + 1

        breeds = features["Breed"].str.split(pat="/", expand=True)[0]
        breeds = pd.concat([breeds, num_breeds], axis=1)
        breeds.columns = ["First_breed", "Num_breeds"]

        # Just in case
        breeds["First_breed"] = breeds["First_breed"].fillna("No")

        return breeds

    def fit(self, features: pd.DataFrame):
        features = features.copy()
        breeds = self.process_breeds(features)

        mapping = breeds.groupby("First_breed")["First_breed"].count()
        self.mapping = mapping[mapping > RARE_THRESHOLD].to_dict()

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.copy()
        breeds = self.process_breeds(features)

        cur_mapping = self.mapping.copy()
        for clr in breeds["First_breed"].unique():
            if clr not in cur_mapping:
                cur_mapping[clr] = 1

        breeds = breeds.replace({"First_breed": cur_mapping})

        features = pd.concat([features, breeds], axis=1)
        features.drop(columns="Breed", inplace=True)

        return features
