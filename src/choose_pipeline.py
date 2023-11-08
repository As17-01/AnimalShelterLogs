from typing import Any
from typing import Dict
from typing import List

import omegaconf
import optuna
from loguru import logger


def _choose_boosting(trial: optuna.Trial, config: omegaconf.DictConfig) -> Dict[str, Any]:
    model_config = {
        "_target_": "sklearn.ensemble.GradientBoostingClassifier",
    }

    model_config["n_estimators"] = trial.suggest_int("n_estimators", 25, 400, 25)
    model_config["max_depth"] = trial.suggest_int("max_depth", 3, 15)
    model_config["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 3, 500)
    model_config["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 1, log=True)

    return model_config


def _choose_weights(trial: optuna.Trial, config: omegaconf.DictConfig) -> List[float]:
    weights: List[float] = []

    weights.append(trial.suggest_float("first", 0.001, 1, log=False))
    weights.append(trial.suggest_float("second", 0.001, 1, log=False))
    weights.append(trial.suggest_float("third", 0.001, 1, log=False))
    weights.append(trial.suggest_float("fourth", 0.001, 1, log=False))
    weights.append(trial.suggest_float("fifth", 0.001, 1, log=False))

    return weights


def choose_pipeline(trial: optuna.Trial, config: omegaconf.DictConfig) -> Dict[str, Any]:
    """Create pipeline config for trial."""
    logger.info("Choosing model")
    model_config = _choose_boosting(trial, config)
    # model_config = {
    #     "_target_": "sklearn.ensemble.GradientBoostingClassifier",
    #     "n_estimators": 200,
    #     "max_depth": 7,
    #     "learning_rate": 0.10707364889452776,
    #     }
    # weights = _choose_weights(trial, config)
    weights = [0.524554751471455, 0.5751646212672618, 0.6912641656697686, 0.9686334173655191, 0.9153012804565787]

    pipeline_config = {
        "_target_": "src.Pipeline",
        "base_model": model_config,
        "weights": weights,
    }
    return pipeline_config
