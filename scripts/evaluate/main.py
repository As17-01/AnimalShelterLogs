import pathlib
import sys
import warnings
import numpy as np
import zipfile
import tempfile
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

sys.path.append("../../")

import src

warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated")
warnings.filterwarnings("ignore", message="is_sparse is deprecated")

TIME = "DateTime"
FEATURES = ["Name", "SexuponOutcome", "AnimalType", "AgeuponOutcome", "Breed", "Color"]
TARGET = "Outcome"


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_path = pathlib.Path(cfg.data.load_key)

    zf = zipfile.ZipFile(load_path)
    with tempfile.TemporaryDirectory() as _temp_dir:
        zf.extractall(_temp_dir)

        temp_dir = pathlib.Path(_temp_dir)
        data = pd.read_csv(temp_dir / cfg.data.train_file_name)

    np.random.seed(120)
    test = data.iloc[np.random.choice(range(len(data)), 5000)]

    vote_list = []
    for p_key in cfg.data.pipeline_keys:
        pipeline = src.Pipeline.load(pathlib.Path(p_key))
        vote_list.append(pipeline.predict(time_index=test[TIME], features=test[FEATURES]))

    outcome = []
    num_voters = len(cfg.data.pipeline_keys)
    for record_id, _ in enumerate(vote_list[0]):
        votes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for vote_id in range(num_voters):
            votes[vote_list[vote_id][record_id]] += 1
        outcome.append(max(votes, key=votes.get))

    logger.info(f"F1: {f1_score(y_true=test[TARGET], y_pred=outcome, average='macro')}")



    data = pd.read_csv(load_path)




if __name__ == "__main__":
    main()
