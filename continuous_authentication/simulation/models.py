from cv2 import sqrt
import numpy as np
from scipy.linalg import sqrtm

def Euclidean(train: np.ndarray, test: np.ndarray, threshold: float) -> np.ndarray:
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
    # sum_diffs = np.sqrt(sum_squared_diffs)
    imposter_vector = np.where(sum_squared_diffs > threshold, 0, 1)

    return imposter_vector

def z_score(train: np.ndarray, test: np.ndarray, threshold: float) -> np.ndarray:
    '''
    compare the zscore to 1.69
    '''
    mean = np.mean(train, axis = 0)
    #list of std
    std = np.std(train, axis = 0, ddof = 1)
    #list of zscore
    zscore = np.absolute((test - mean)/std)
    vec_labels = np.where(zscore<threshold, 1,0)
    thresh2 = .5
    test_sums = np.sum(vec_labels, axis = 1)
    test_size = test.shape[1]
    ratio = test_sums/test_size
    pred_labels = np.where(ratio<thresh2, 0, 1)

    return pred_labels

def Manhattan(train: np.ndarray, test: np.ndarray) -> np.ndarray:
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
    abs_diffs = np.absolute(diffs)
    sum_diffs = np.sum(abs_diffs, axis = 1).tolist()

    return sum_diffs

def Scaled_Manhattan(train: np.ndarray, test: np.ndarray) -> np.ndarray:
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
    std = np.std(train, axis = 0)
    diffs = model[np.newaxis, :] - test
    abs_diffs = np.absolute(diffs) / std # optionally mad instead of std, but std performs marginally better
    sum_diffs = np.sum(abs_diffs, axis = 1)

    return sum_diffs

def Zhong_Deng(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    S_inv_sqrt = np.linalg.inv(sqrtm(np.cov(train, rowvar = False))) 
    diffs = train - test
    trans_diffs = S_inv_sqrt * diffs
    abs_diffs = np.absolute(trans_diffs.real)
    total = np.sum(abs_diffs)

    # sqrt_x_cov = sqrtm(x_cov)
    # x_cov_atmpt = sqrt_x_cov @ sqrt_x_cov
    # inv_x_sqrt = np.linalg.inv(sqrt_x_cov)

    # print(f"Covariance Matrix: \n{x_cov}")
    # print(f"\n\Real Square Root of Cov. Matrix: \n{sqrt_x_cov}")
    # print(f"\n\nAttempted Covariance Matrix: \n{x_cov_atmpt.real}\n")
    # print(f"\n\nInverse of Principal SQRT of Cov. Matrix: \n{inv_x_sqrt}")

    # inv_x_sqrt = np.linalg.inv(sqrt_x_cov)

    return total

if __name__== "__main__":
    train = np.array([2, 1, 2])
    test = np.array([2, 1, 34])
    print(Zhong_Deng(train, test))