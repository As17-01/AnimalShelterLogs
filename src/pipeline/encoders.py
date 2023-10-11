from typing import Dict

import numpy as np
import pandas as pd


class NameEncoder:
    def __init__(self):
        self.mapping: Dict[str, int] = {}

    def fit(self, features: pd.DataFrame):
        features = features.copy()

        features["Name"] = features["Name"].str.replace(" ", "")
        self.mapping = features.groupby("Name")["Name"].count().to_dict()

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

        colors = features["Color"].str.split(pat="/", expand=True)
        colors = pd.concat([colors, num_colors], axis=1)
        colors.columns = ["First_color", "Second_color", "Num_colors"]

        colors["Is_complex_color"] = colors["First_color"].str.count(" ") + colors["Second_color"].str.count(" ")

        colors[["First_color", "Second_color"]] = colors[["First_color", "Second_color"]].fillna("No")

        return colors

    def fit(self, features: pd.DataFrame):
        features = features.copy()
        colors = self.process_colors(features)

        melted_colors = colors[["First_color", "Second_color"]].melt(value_name="color")
        self.mapping = melted_colors.groupby("color")["color"].count().to_dict()

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.copy()
        colors = self.process_colors(features)

        cur_mapping = self.mapping.copy()
        for clr in colors["First_color"].unique():
            if clr not in cur_mapping:
                cur_mapping[clr] = 1
        for clr in colors["Second_color"].unique():
            if clr not in cur_mapping:
                cur_mapping[clr] = 1

        colors = colors.replace({"First_color": cur_mapping})
        colors = colors.replace({"Second_color": cur_mapping})

        sorted_colors = pd.DataFrame(np.sort(colors.values, axis=1), columns=colors.columns)
        colors["First_color"] = sorted_colors["First_color"]
        colors["Second_color"] = sorted_colors["Second_color"]

        features = pd.concat([features, colors], axis=1)
        features.drop(columns="Color", inplace=True)

        return features


class BreedEncoder:
    def __init__(self):
        self.mapping: Dict[str, int] = {}

    @staticmethod
    def process_breeds(features: pd.DataFrame) -> pd.DataFrame:
        num_breeds = features["Breed"].str.count("/") + 1

        breeds = features["Breed"].str.split(pat="/", expand=True)
        breeds = pd.concat([breeds, num_breeds], axis=1)
        breeds.columns = ["First_breed", "Second_breed", "Third_breed", "Num_breeds"]

        breeds[["First_breed", "Second_breed", "Third_breed"]] = breeds[
            ["First_breed", "Second_breed", "Third_breed"]
        ].fillna("No")

        return breeds

    def fit(self, features: pd.DataFrame):
        features = features.copy()
        breeds = self.process_breeds(features)

        melted_breeds = breeds[["First_breed", "Second_breed", "Third_breed"]].melt(value_name="breed")
        self.mapping = melted_breeds.groupby("breed")["breed"].count().to_dict()

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.copy()
        breeds = self.process_breeds(features)

        cur_mapping = self.mapping.copy()
        for clr in breeds["First_breed"].unique():
            if clr not in cur_mapping:
                cur_mapping[clr] = 1
        for clr in breeds["Second_breed"].unique():
            if clr not in cur_mapping:
                cur_mapping[clr] = 1
        for clr in breeds["Third_breed"].unique():
            if clr not in cur_mapping:
                cur_mapping[clr] = 1

        breeds = breeds.replace({"First_breed": cur_mapping})
        breeds = breeds.replace({"Second_breed": cur_mapping})
        breeds = breeds.replace({"Third_breed": cur_mapping})

        sorted_breeds = pd.DataFrame(np.sort(breeds.values, axis=1), columns=breeds.columns)
        breeds["First_breed"] = sorted_breeds["First_breed"]
        breeds["Second_breed"] = sorted_breeds["Second_breed"]
        breeds["Third_breed"] = sorted_breeds["Third_breed"]

        features = pd.concat([features, breeds], axis=1)
        features.drop(columns="Breed", inplace=True)

        return features
