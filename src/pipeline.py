import pathlib
import pickle
import tempfile
import zipfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score

from src.encoders import CategoricalEncoder
from src.encoders import process_breeds
from src.encoders import process_colors
from src.encoders import process_names
from src.utils import process_features
from src.utils import process_time


class Pipeline:
    def __init__(self, base_model: Any, oversampler: Optional[Any]):
        self.base_model = base_model
        self.oversampler = oversampler

        self.weights: List[int] = []

        self.name_encoder = CategoricalEncoder(col_name="Name", callback=process_names, rare_threshold=10)
        self.color_encoder = CategoricalEncoder(col_name="Color", callback=process_colors, rare_threshold=30)
        self.breed_encoder = CategoricalEncoder(col_name="Breed", callback=process_breeds, rare_threshold=25)

    def prepare_data(self, time_index: pd.Series, features=pd.DataFrame) -> pd.DataFrame:
        # logger.info(f"Processing features")
        features = process_features(features=features)
        features = self.name_encoder.encode(features=features)
        features = self.color_encoder.encode(features=features)
        features = self.breed_encoder.encode(features=features)

        # logger.info(f"Formating datetime")
        features = process_time(features=features, time=time_index)

        return features

    def optimize_weights(self, features: pd.DataFrame, target=pd.Series):
        probs = self.base_model.predict_proba(features)

        # TODO: Rebalance instead
        self.weights = [6.342, 4.484, 7.036, 8.988, 9.327]
        # predictions = np.argmax(probs * np.array(self.weights), axis=1)
        # best_f1 = f1_score(y_true=target, y_pred=predictions, average="macro")
        # for i in range(0, 1000):
        #     candidate = np.random.uniform(0, 10, 5).tolist()

        #     predictions = np.argmax(probs * np.array(candidate), axis=1)
        #     f1 = f1_score(y_true=target, y_pred=predictions, average='macro')
        #     if f1 > best_f1:
        #         best_f1 = f1
        #         self.weights = candidate
        # logger.info(self.weights)

    def fit(self, time_index: pd.Series, features=pd.DataFrame, target=pd.Series):
        # logger.info(f"Fitting encoders")
        self.name_encoder.fit(features=features)
        self.color_encoder.fit(features=features)
        self.breed_encoder.fit(features=features)

        # logger.info(f"Preparing data")
        features = self.prepare_data(time_index=time_index, features=features)

        # if self.oversampler:
        #     # logger.info(f"Oversampling data")
        #     new_features, new_target = self.oversampler.resample(features=features, target=target)

        # logger.info(f"Starting train")
        self.base_model.fit(features, target)

        # logger.info(f"Optimizing weights")
        self.optimize_weights(features, target)

    def predict(self, time_index: pd.Series, features=pd.DataFrame) -> pd.Series:
        # logger.info(f"Preparing data")
        features = self.prepare_data(time_index=time_index, features=features)

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
                pickle.dump(self.base_model, open(output_path / f"oversampler.pkl", "wb"))
                pickle.dump(self.name_encoder, open(output_path / f"name_encoder.pkl", "wb"))
                pickle.dump(self.color_encoder, open(output_path / f"color_encoder.pkl", "wb"))
                pickle.dump(self.breed_encoder, open(output_path / f"breed_encoder.pkl", "wb"))

                archieve.write(output_path / f"par.pkl", pathlib.Path(f"par.pkl"))
                archieve.write(output_path / f"weights.pkl", pathlib.Path(f"weights.pkl"))
                archieve.write(output_path / f"base_model.pkl", pathlib.Path(f"base_model.pkl"))
                archieve.write(output_path / f"oversampler.pkl", pathlib.Path(f"oversampler.pkl"))
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
            weights = pickle.load(open(output_path / f"weights.pkl", "rb"))
            base_model = pickle.load(open(output_path / f"base_model.pkl", "rb"))
            oversampler = pickle.load(open(output_path / f"oversampler.pkl", "rb"))
            name_encoder = pickle.load(open(output_path / f"name_encoder.pkl", "rb"))
            color_encoder = pickle.load(open(output_path / f"color_encoder.pkl", "rb"))
            breed_encoder = pickle.load(open(output_path / f"breed_encoder.pkl", "rb"))

        loaded_instance = cls(base_model=base_model, oversampler=oversampler, **par)
        loaded_instance.weights = weights
        loaded_instance.name_encoder = name_encoder
        loaded_instance.color_encoder = color_encoder
        loaded_instance.breed_encoder = breed_encoder
        return loaded_instance
