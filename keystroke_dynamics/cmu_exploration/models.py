import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import *


def Euclidean_Distance(train: np.ndarray, test: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculate arithmetic mean of train along axis 1 (model)
    Then compute pairwise distances between test entries and model
    Distances below threshold are cast to 1's (real user) and above threshold are 0's (imposter)

    Args: 
        train: data to form model from
        test: data to compare to model vector
        threshold: barrier between authentic user and imposter

    Returns: vector of length = len(test)
    """
    model = np.mean(train, axis = 0)
    diffs = model[np.newaxis, :] - test
    squared_diffs = np.square(diffs)
    sum_squared_diffs = np.sum(squared_diffs, axis = 1)
    sum_diffs = np.sqrt(sum_squared_diffs)
    imposter_vector = np.where(sum_diffs > threshold, 0, 1)

    return imposter_vector