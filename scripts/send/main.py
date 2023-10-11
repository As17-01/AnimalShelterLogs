import pathlib
import sys

import hydra
import kaggle
from omegaconf import DictConfig

sys.path.append("../../")


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    data_path = pathlib.Path(cfg.data.load_key)

    client = kaggle.ApiClient()
    api = kaggle.KaggleApi(client)

    api.authenticate()
    api.competition_submit(
        file_name=data_path / cfg.data.file_name,
        message=cfg.data.submit_message,
        competition="animal-shelter-logs",
    )


if __name__ == "__main__":
    main()
