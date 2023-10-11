import pathlib
import pickle
import tempfile
import zipfile
from enum import Enum
from enum import auto
from typing import Any

import pandas as pd
from loguru import logger

from src.pipeline.encoders import BreedEncoder
from src.pipeline.encoders import ColorEncoder
from src.pipeline.encoders import NameEncoder
from src.pipeline.utils import process_features


class BaseMode(Enum):
    """Enum for mode."""

    reg = auto()
    clas = auto()

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} types are allowed"
        )


class Pipeline:
    def __init__(self, base_model: Any, mode: str):
        self.base_model = base_model
        self.mode = mode

        self.name_encoder = NameEncoder()
        self.color_encoder = ColorEncoder()
        self.breed_encoder = BreedEncoder()

    def fit(self, time_index: pd.Series, features=pd.DataFrame, target=pd.Series):
        logger.info(f"Fitting encoders")
        self.name_encoder.fit(features)
        self.color_encoder.fit(features)
        self.breed_encoder.fit(features)

        logger.info(f"Processing features")
        features = process_features(features)
        features = self.name_encoder.encode(features)
        features = self.color_encoder.encode(features)
        features = self.breed_encoder.encode(features)

        logger.info(f"Formating datatypes")
        for feature_name in features.columns:
            if feature_name != "num_age":
                features[feature_name] = features[feature_name].astype("category")

        logger.info(f"Starting train")
        self.base_model.fit(features, target)

    def predict(self, time_index: pd.Series, features=pd.DataFrame) -> pd.Series:
        logger.info(f"Processing features")
        features = process_features(features)
        features = self.name_encoder.encode(features)
        features = self.color_encoder.encode(features)
        features = self.breed_encoder.encode(features)

        logger.info(f"Formating datatypes")
        for feature_name in features.columns:
            if feature_name != "num_age":
                features[feature_name] = features[feature_name].astype("category")

        logger.info(f"Making predictions")
        predictions = self.base_model.predict(features)

        predictions = pd.Series(predictions)
        return predictions

    def save(self, path: pathlib.Path):
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archieve:
                par = {"mode": self.mode}

                pickle.dump(par, open(output_path / f"par.pkl", "wb"))
                pickle.dump(self.base_model, open(output_path / f"base_model.pkl", "wb"))
                pickle.dump(self.name_encoder, open(output_path / f"name_encoder.pkl", "wb"))
                pickle.dump(self.color_encoder, open(output_path / f"color_encoder.pkl", "wb"))
                pickle.dump(self.breed_encoder, open(output_path / f"breed_encoder.pkl", "wb"))

                archieve.write(output_path / f"par.pkl", pathlib.Path(f"par.pkl"))
                archieve.write(output_path / f"base_model.pkl", pathlib.Path(f"base_model.pkl"))
                archieve.write(output_path / f"name_encoder.pkl", pathlib.Path(f"name_encoder.pkl"))
                archieve.write(output_path / f"color_encoder.pkl", pathlib.Path(f"color_encoder.pkl"))
                archieve.write(output_path / f"breed_encoder.pkl", pathlib.Path(f"breed_encoder.pkl"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "Pipeline":
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="r") as archieve:
                archieve.extractall(output_path)

            par = pickle.load(open(output_path / f"par.pkl", "rb"))
            base_model = pickle.load(open(output_path / f"base_model.pkl", "rb"))
            name_encoder = pickle.load(open(output_path / f"name_encoder.pkl", "rb"))
            color_encoder = pickle.load(open(output_path / f"color_encoder.pkl", "rb"))
            breed_encoder = pickle.load(open(output_path / f"breed_encoder.pkl", "rb"))

        loaded_instance = cls(base_model=base_model, **par)
        loaded_instance.name_encoder = name_encoder
        loaded_instance.color_encoder = color_encoder
        loaded_instance.breed_encoder = breed_encoder
        return loaded_instance
