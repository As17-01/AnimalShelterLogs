from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd


class Oversampler:
    def __init__(self, baseline_class: int = 0, random_state: Optional[int] = None):
        self.baseline_class = baseline_class
        self.random_state = random_state

    def resample(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        np.random.seed(self.random_state)
        features = features.copy()
        features["target"] = target

        splitted_by_target = [
            features[target == 0],
            features[target == 1],
            features[target == 2],
            features[target == 3],
            features[target == 4],
        ]
        req_size = len(splitted_by_target[self.baseline_class])

        sampled = pd.concat([split.iloc[np.random.choice(len(split), req_size)] for split in splitted_by_target])

        target = sampled["target"]
        sampled.drop(columns="target", inplace=True)
        return sampled, target
