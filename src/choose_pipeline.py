import optuna
from typing import Dict, Any, List
import omegaconf
from loguru import logger

def _choose_xgb(trial: optuna.Trial, config: omegaconf.DictConfig) -> Dict[str, Any]:
    model_config = {
        "_target_": "xgboost.XGBClassifier",
        "n_estimators": 300,  # Let's only choose the best learning rate with the fixed number of iterations
        "enable_categorical": True,
    }

    model_config["max_depth"] = trial.suggest_int("max_depth", 1, 15)
    model_config["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 1, log=True)
    model_config["gamma"] = trial.suggest_float("gamma", 1, 9, log=True),
    model_config["reg_alpha"] = trial.suggest_int("reg_alpha", 40, 180, 1),
    model_config["reg_lambda"] = trial.suggest_float("reg_lambda", 0, 1, log=True),
    model_config["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1, log=True),
    model_config["min_child_weight"] = trial.suggest_int("min_child_weight", 0, 10, 1),

    return model_config


def _choose_weights(trial: optuna.Trial, config: omegaconf.DictConfig) -> Dict[str, Any]:
    weights: List[float] = []

    weights.append(trial.suggest_float("first", 0.001, 1, log=True))
    weights.append(trial.suggest_float("second", 0.001, 1, log=True))
    weights.append(trial.suggest_float("third", 0.001, 1, log=True))
    weights.append(trial.suggest_float("fourth", 0.001, 1, log=True))
    weights.append(trial.suggest_float("fifth", 0.001, 1, log=True))

    return weights

def choose_pipeline(trial: optuna.Trial, config: omegaconf.DictConfig) -> Dict[str, Any]:
    """Create pipeline config for trial."""
    logger.info("Choosing model")
    model_config = _choose_xgb(trial, config)
    weights = _choose_weights(trial, config)

    pipeline_config = {
        "_target_": "src.Pipeline",
        "base_model": model_config,
        "weights": weights,
    }
    return pipeline_config