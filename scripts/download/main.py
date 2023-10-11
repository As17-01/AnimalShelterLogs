import pathlib
import sys

import hydra
import kaggle
from omegaconf import DictConfig

sys.path.append("../../")


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    data_path = pathlib.Path(cfg.data.save_key)

    client = kaggle.ApiClient()
    api = kaggle.KaggleApi(client)

    api.authenticate()
    api.competition_download_file(
        file_name=cfg.data.file_name,
        path=data_path,
        competition="animal-shelter-logs",
    )


if __name__ == "__main__":
    main()
