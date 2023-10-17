from typing import Any
from typing import Dict
from typing import List

import omegaconf
import optuna
from loguru import logger


def _choose_xgb(trial: optuna.Trial, config: omegaconf.DictConfig) -> Dict[str, Any]:
    model_config = {
        "_target_": "xgboost.XGBClassifier",
        "n_estimators": 200,  # Let's only choose the best learning rate with the fixed number of iterations
        "enable_categorical": True,
    }

    model_config["max_depth"] = trial.suggest_int("max_depth", 3, 15)
    model_config["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 1, log=True)
    model_config["gamma"] = trial.suggest_float("gamma", 0, 9, log=False)
    # model_config["reg_alpha"] = trial.suggest_int("reg_alpha", 0, 180, 1)
    # model_config["reg_lambda"] = trial.suggest_float("reg_lambda", 0, 1, log=False)
    # model_config["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1, log=False)
    # model_config["min_child_weight"] = trial.suggest_int("min_child_weight", 0, 10, 1)

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
    model_config = _choose_xgb(trial, config)
    # model_config = {
    #     "_target_": "xgboost.XGBClassifier",
    #     "n_estimators": 200,
    #     "enable_categorical": True,
    #     "max_depth": 8,
    #     "learning_rate": 0.004,
    #     "gamma": 0.0004,
    #     }
    # weights = _choose_weights(trial, config)
    weights = [0.524554751471455, 0.5751646212672618, 0.6912641656697686, 0.9686334173655191, 0.9153012804565787]

    pipeline_config = {
        "_target_": "src.Pipeline",
        "base_model": model_config,
        "weights": weights,
    }
    return pipeline_config
