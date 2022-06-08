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

def z_score (train: np.ndarray, test: np.ndarray, threshold: float) -> np.ndarray:
    '''
    compair the zscore to 1.69
    '''
    mean = np.mean(train, axis = 0)
    print("mean: ",mean)

#list of std
    std = np.std(train, axis = 0, ddof = 1)
    print("std: ",std)
    #list of zscore
    zscore = np.absolute((test - mean)/std)
    
    vec_lables = np.where(zscore<threshold, 1,0)
    
    thresh2 = .5


    test_sums = np.sum(vec_lables, axis = 1)
    

    test_size = synth_test.shape
    

    ratio = test_sums/test_size
    

    pred_lables = np.where(ratio<thresh2, 0, 1)

    return pred_lables