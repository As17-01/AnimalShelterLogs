import pathlib
import sys
import pandas as pd

import hydra
from omegaconf import DictConfig

sys.path.append("../../")

import src
import src.pipeline

TIME = "DateTime"
FEATURES = ["Name", "SexuponOutcome", "AnimalType", "AgeuponOutcome", "Breed", "Color"]

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_path = pathlib.Path(cfg.data.load_key)
    save_path = pathlib.Path(cfg.data.save_key)
    pipeline_key = pathlib.Path(cfg.data.pipeline_key)

    data = pd.read_csv(load_path)

    pipeline = src.pipeline.Pipeline.load(pipeline_key)
    data["Outcome"] = pipeline.predict(time_index=data[TIME], features=data[FEATURES])

    data = data[["ID", "Outcome"]]
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
