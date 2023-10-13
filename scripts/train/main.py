import pathlib
import sys
import tempfile
import warnings
import zipfile

import hydra
import numpy as np
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


def sample_classes(data: pd.DataFrame, mode: str = "min") -> pd.DataFrame:
    sampled = data.copy()

    zeros = sampled[sampled[TARGET] == 0]
    ones = sampled[sampled[TARGET] == 1]
    twos = sampled[sampled[TARGET] == 2]
    threes = sampled[sampled[TARGET] == 3]
    fours = sampled[sampled[TARGET] == 4]

    if mode == "min":
        req_size = len(fours)
    elif mode == "max":
        req_size = len(zeros)
    else:
        raise ValueError("Which mode did you choose?")

    zeros = zeros.iloc[np.random.choice(len(ones), req_size)]
    logger.info(f"Size of zeros: {len(zeros)}")

    ones = ones.iloc[np.random.choice(len(ones), req_size)]
    logger.info(f"Size of ones: {len(ones)}")

    twos = twos.iloc[np.random.choice(len(twos), req_size)]
    logger.info(f"Size of twos: {len(twos)}")

    threes = threes.iloc[np.random.choice(len(threes), req_size)]
    logger.info(f"Size of threes: {len(threes)}")

    fours = fours.iloc[np.random.choice(len(fours), req_size)]
    logger.info(f"Size of fours: {len(fours)}")

    sampled = pd.concat([zeros, ones, twos, threes, fours])
    return sampled


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

    # They are NOT splitted by time. I can include TIME as a feature
    kf = KFold(n_splits=5, shuffle=True, random_state=100)

    # python main.py pipeline=xgb_base data=xgb_base
    for i, (train_index, val_index) in enumerate(kf.split(data, data[TARGET])):
        logger.info(f"Fold {i}")
        train = data.iloc[train_index]
        logger.info(f"Train size: {len(train)}")
        val = data.iloc[val_index]
        logger.info(f"Val size: {len(val)}")

        # logger.info(f"Sampling classes")
        # train = sample_classes(data=train)
        # logger.info(f"Train size after sampling: {len(train)}")

        pipeline.fit(time_index=train[TIME], features=train[FEATURES], target=train[TARGET])

        predictions = pipeline.predict(time_index=val[TIME], features=val[FEATURES])
        logger.info(f"Non-averaged F1: {f1_score(y_true=val[TARGET], y_pred=predictions, average=None)}")
        logger.info(f"F1: {f1_score(y_true=val[TARGET], y_pred=predictions, average='macro')}")

        pipeline.save(pipeline_key / (cfg.data.pipeline_name + f"{i}.zip"))


if __name__ == "__main__":
    main()
