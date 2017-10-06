from types import *
from typing import List
import numpy as np
import tensorflow as tf


class Brain:
    def __init__(self):
        pass

    def predict(self, obs: Observation) -> np.ndarray:
        pass

    def train(self, batch: List[Sample], t: int):
        pass
