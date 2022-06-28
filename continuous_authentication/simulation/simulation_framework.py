import numpy as np
from pathlib import PurePath
import os
import sys
sys.path.append("../../")

from continuous_authentication.feature_extraction.parse_utils import *

def get_train(user_arr: np.ndarray, test_digraphs: int) -> tuple:
    rng = np.random.default_rng()
    eof = len(user_arr) 
    # determine the upper bound for where to start recording digraphs
    stop_pos = 0
    digraph_cnt = 0
    for i, val in reversed(list(enumerate(user_arr))):
        digraphs = int(val[2])
        if digraph_cnt + digraphs <= test_digraphs: 
            digraph_cnt += digraphs

        else: 
            stop_pos = i
            break

    random_start_position = rng.choice(np.arange(stop_pos))
    print(f"With {eof} words, start was selected at {random_start_position}")

    return random_start_position

def main(dd):
    # Data import
    default_raw_path = PurePath("../../data/clarkson2_files/")
    default_timeseries_path = PurePath("../../data/user_time_series/")
    user_list = [user for user in os.listdir(default_raw_path) if user != ".DS_Store"]
    path = lambda user: PurePath(default_timeseries_path, PurePath(f"user_{user}.csv"))
    read_paths = [path(user) for user in user_list]

    list_of_user_arrays = [np.genfromtxt(path, dtype = np.dtype(object), delimiter = "\t") for path in read_paths]

    # For each user, perform cross validation
    for i, user_arr in enumerate(list_of_user_arrays):
        # Pull out training data
        train, remainder = get_train(user_arr, test_digraphs = dd), None

        # Generate arrays
        # arrays_by_folds = [user_arr[fold] for fold in folds]

        # Start a simulation with each array
        # results = [simulate(array) for array in arrays_by_folds]

    pass

if __name__ == "__main__":
    main(dd = 100)
