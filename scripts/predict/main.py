import pathlib
import sys
import warnings
from loguru import logger

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.append("../../")

import src
import src.pipeline

warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated")
warnings.filterwarnings("ignore", message="is_sparse is deprecated")

TIME = "DateTime"
FEATURES = ["Name", "SexuponOutcome", "AnimalType", "AgeuponOutcome", "Breed", "Color"]


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_path = pathlib.Path(cfg.data.load_key)
    save_path = pathlib.Path(cfg.data.save_key)

    data = pd.read_csv(load_path)

    logger.info(f"Making predictions")
    vote_list = []
    for p_key in cfg.data.pipeline_keys:
        logger.info(f"Current pipeline: {p_key}")
        pipeline = src.pipeline.Pipeline.load(pathlib.Path(p_key))
        vote_list.append(pipeline.predict(time_index=data[TIME], features=data[FEATURES]))

    logger.info(f"Calculating votes")
    # Maybe regression instead of classification?
    outcome = []
    num_voters = len(cfg.data.pipeline_keys)
    for record_id, _ in enumerate(vote_list[0]):
        votes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for vote_id in range(num_voters):
            votes[vote_list[vote_id][record_id]] += 1
        outcome.append(max(votes, key=votes.get))

    data["Outcome"] = pd.Series(outcome)

    logger.info(f"Saving data to {save_path}")
    data = data[["ID", "Outcome"]]
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
