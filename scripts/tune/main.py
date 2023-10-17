import pathlib
import sys
import tempfile
import warnings
import zipfile

import hydra
import omegaconf
import optuna
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


def objective(
    trial: optuna.Trial,
    data: pd.DataFrame,
    registry: Registry,
    config: omegaconf.DictConfig,
) -> float:
    """Optuna objective."""
    logger.info("Forming config for cross-validation")
    pipeline_config = src.choose_pipeline(trial, config)
    pipeline = registry.get_from_params(**pipeline_config)
    logger.info(f"Selected config: {trial.params}")

    kf = KFold(n_splits=4, shuffle=True, random_state=100)

    metric_history = []
    for i, (train_index, val_index) in enumerate(kf.split(data, data[TARGET])):
        train = data.iloc[train_index]
        val = data.iloc[val_index]

        pipeline.fit(time_index=train[TIME], features=train[FEATURES], target=train[TARGET])

        predictions = pipeline.predict(time_index=val[TIME], features=val[FEATURES])

        metric_history.append(f1_score(y_true=val[TARGET], y_pred=predictions, average="macro"))

    return sum(metric_history) / len(metric_history)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    load_path = pathlib.Path(cfg.data.load_key)

    registry = Registry()
    registry.add_from_module(src, prefix="src.")

    zf = zipfile.ZipFile(load_path)
    with tempfile.TemporaryDirectory() as _temp_dir:
        zf.extractall(_temp_dir)

        temp_dir = pathlib.Path(_temp_dir)
        data = pd.read_csv(temp_dir / cfg.data.train_file_name)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        study_name="optimization",
    )
    logger.info("Start hyperparams optimization")
    study.optimize(
        lambda trial: objective(trial, data, registry, cfg),
        catch=(Exception,),
        gc_after_trial=True,
        n_trials=100,
    )


if __name__ == "__main__":
    main()
