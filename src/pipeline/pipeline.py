import pathlib
import pickle
import tempfile
import zipfile
from typing import Any

import pandas as pd


class Pipeline:
    def __init__(self, model: Any):
        self.model = model

    def fit(self, time_index: pd.DataFrame, features=pd.DataFrame, target=pd.DataFrame):

        self.model.fit(features, target)

    def predict(self, time_index: pd.DataFrame, features=pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(features)

        predictions = pd.Series(predictions)
        return predictions

    def save(self, path: pathlib.Path):
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archieve:
                pickle.dump(self.model, open(output_path / f"model.pkl", "wb"))

                archieve.write(output_path / f"model.pkl", pathlib.Path(f"model.pkl"))

    @classmethod
    def load(cls, path: pathlib.Path) -> "Pipeline":
        with tempfile.TemporaryDirectory() as _output_path:
            output_path = pathlib.Path(_output_path)

            with zipfile.ZipFile(path, mode="r") as archieve:
                archieve.extractall(output_path)

            model = pickle.load(open(output_path / f"model.pkl", "rb"))

        loaded_instance = cls(model=model)
        return loaded_instance
