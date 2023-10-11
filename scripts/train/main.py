import pathlib
import sys
import tempfile
import warnings
import zipfile
from loguru import logger

import hydra
import omegaconf
import pandas as pd
from hydra_slayer import Registry
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

sys.path.append("../../")

import src
import src.pipeline

warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated")
warnings.filterwarnings("ignore", message="is_sparse is deprecated")

TIME = "DateTime"
FEATURES = ["Name", "SexuponOutcome", "AnimalType", "AgeuponOutcome", "Breed", "Color"]
TARGET = "Outcome"


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    load_path = pathlib.Path(cfg.data.load_key)
    pipeline_key = pathlib.Path(cfg.data.pipeline_key)

    zf = zipfile.ZipFile(load_path)
    with tempfile.TemporaryDirectory() as _temp_dir:
        zf.extractall(_temp_dir)

        temp_dir = pathlib.Path(_temp_dir)
        data = pd.read_csv(temp_dir / cfg.data.train_file_name)

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()

    registry.add_from_module(src.pipeline, prefix="src.pipeline.")
    pipeline = registry.get_from_params(**cfg_dct["pipeline"])

    kf = TimeSeriesSplit(n_splits=4)

    for i, (train_index, val_index) in enumerate(kf.split(data, data[TARGET])):
        logger.info(f"Fold {i}")
        train = data.iloc[train_index]
        val = data.iloc[val_index]

        pipeline.fit(time_index=train[TIME], features=train[FEATURES], target=train[TARGET])

        predictions = pipeline.predict(time_index=val[TIME], features=val[FEATURES])
        logger.info(f"ACC: {accuracy_score(y_true=val[TARGET], y_pred=predictions)}")
    
        pipeline.save(pipeline_key)


if __name__ == "__main__":
    main()
