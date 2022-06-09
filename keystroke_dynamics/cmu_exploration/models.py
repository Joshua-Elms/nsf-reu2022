import numpy as np

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


def Manhattan(train: np.ndarray, test: np.ndarray, threshold: float) -> np.ndarray:
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
    sum_diffs = np.sum(abs_diffs, axis = 1)
    imposter_vector = np.where(sum_diffs > threshold, 0, 1)

    return imposter_vector

def Scaled_Manhattan(train: np.ndarray, test: np.ndarray, threshold: float) -> np.ndarray:
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
    mad0 = train - model
    mad1 = np.absolute(mad0)
    mad2 = np.sum(mad1, axis = 0)
    mad3 = mad2 / train.shape[0]
    mad = mad3[np.newaxis, :]
    diffs = model[np.newaxis, :] - test
    abs_diffs = np.absolute(diffs) / mad
    sum_diffs = np.sum(abs_diffs, axis = 1)
    imposter_vector = np.where(sum_diffs > threshold, 0, 1)

    return imposter_vector


if __name__== "__main__":
    train = np.array([[2, 1], [1, 3]])
    test = np.array([[1, 3], [2, 1]])
    thresh = 2
    print(Scaled_Manhattan(train, test, thresh))