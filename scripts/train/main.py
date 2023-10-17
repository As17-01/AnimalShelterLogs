import pathlib
import sys
import tempfile
import warnings
import zipfile

import hydra
import omegaconf
import pandas as pd
from hydra_slayer import Registry
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

sys.path.append("../../")

import src

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

    registry.add_from_module(src, prefix="src.")
    pipeline = registry.get_from_params(**cfg_dct["pipeline"])

    kf = KFold(n_splits=4, shuffle=True, random_state=100)

    metric_history = []
    for i, (train_index, val_index) in enumerate(kf.split(data, data[TARGET])):
        logger.info(f"Fold {i}")

        train = data.iloc[train_index]
        val = data.iloc[val_index]

        pipeline.fit(time_index=train[TIME], features=train[FEATURES], target=train[TARGET])

        predictions = pipeline.predict(time_index=val[TIME], features=val[FEATURES])

        pipeline.save(pipeline_key / (cfg.data.pipeline_name + f"{i}.zip"))

        metric_history.append(f1_score(y_true=val[TARGET], y_pred=predictions, average="macro"))

    logger.info(f"F1: {sum(metric_history) / len(metric_history)}")


if __name__ == "__main__":
    main()
