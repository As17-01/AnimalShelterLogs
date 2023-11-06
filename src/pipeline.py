import pathlib
import pickle
import tempfile
import zipfile
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import process_features
from src.utils import process_time


class Pipeline:
    def __init__(self, base_model: Any, weights: List[float]):
        self.base_model = base_model
        self.weights = weights
        self.encoders = []

    def prepare_data(self, time_index: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        # logger.info(f"Processing features")
        features = process_features(features=features)

        # Color is useless
        features.drop(columns="Color", inplace=True)

        # Breed is useless
        features.drop(columns="Breed", inplace=True)

        # Name is useless
        features.drop(columns="Name", inplace=True)

        # logger.info(f"Formating datetime")
        features = process_time(features=features, time=time_index)

        return features

    def fit(self, time_index: pd.Series, features=pd.DataFrame, target=pd.Series):
        # logger.info(f"Preparing data")
        features = self.prepare_data(time_index=time_index, features=features)

        cat_columns = [col_name for col_name in features.columns if features[col_name].dtypes == "category"]
        for i, col_name in enumerate(cat_columns):
            le = LabelEncoder()
            features[col_name] = le.fit_transform(features[col_name])
            self.encoders.append(le)

        # logger.info(f"Starting train")
        self.base_model.fit(features, target)

    def predict(self, time_index: pd.Series, features=pd.DataFrame) -> pd.Series:
        # logger.info(f"Preparing data")
        features = self.prepare_data(time_index=time_index, features=features)

        cat_columns = [col_name for col_name in features.columns if features[col_name].dtypes == "category"]
        for i, col_name in enumerate(cat_columns):
            features[col_name] = self.encoders[i].transform(features[col_name])

        # logger.info(f"Making predictions")
        probs = self.base_model.predict_proba(features)
        predictions = np.argmax(probs * np.array(self.weights), axis=1)
        predictions = pd.Series(predictions.ravel()).astype(int)

        return predictions

    def save(self, path: pathlib.Path):
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archieve:
                par: Dict[str, Any] = {}

                pickle.dump(par, open(output_path / f"par.pkl", "wb"))
                pickle.dump(self.weights, open(output_path / f"weights.pkl", "wb"))
                pickle.dump(self.base_model, open(output_path / f"base_model.pkl", "wb"))
                pickle.dump(self.encoders, open(output_path / f"encoders.pkl", "wb"))

                archieve.write(output_path / f"par.pkl", pathlib.Path(f"par.pkl"))
                archieve.write(output_path / f"weights.pkl", pathlib.Path(f"weights.pkl"))
                archieve.write(output_path / f"base_model.pkl", pathlib.Path(f"base_model.pkl"))
                archieve.write(output_path / f"encoders.pkl", pathlib.Path(f"encoders.pkl"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "Pipeline":
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="r") as archieve:
                archieve.extractall(output_path)

            par = pickle.load(open(output_path / f"par.pkl", "rb"))
            weights = pickle.load(open(output_path / f"weights.pkl", "rb"))
            base_model = pickle.load(open(output_path / f"base_model.pkl", "rb"))
            encoders = pickle.load(open(output_path / f"encoders.pkl", "rb"))

        loaded_instance = cls(base_model=base_model, weights=weights, **par)
        loaded_instance.encoders = encoders
        return loaded_instance
