from enum import Enum

from sklearn.base import BaseEstimator


class BaseModelType(Enum):
    """Enum for models."""

    sklearn = BaseEstimator

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} types are allowed"
        )