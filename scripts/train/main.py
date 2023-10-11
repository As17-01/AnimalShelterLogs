import pathlib
import sys
import pandas as pd
import zipfile
import tempfile
from hydra_slayer import Registry

import hydra
import omegaconf

sys.path.append("../../")

import src
import src.pipeline


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

    pipeline.fit(time_index=data[TIME], features=data[FEATURES], target=data[TARGET])
    pipeline.save(pipeline_key)


if __name__ == "__main__":
    main()
