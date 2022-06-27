import numpy as np
from pathlib import PurePath
import os
from json import load
import sys
sys.path.append("../../")

from continuous_authentication.feature_extraction.parse_utils import *

def read_in_data(path_lst: list) -> tuple:
    all_users_raw_nested_tuples = [file_to_nested_tuples(path) for path in path_lst]
    all_users_clean_nested_tuples = [clean_text(dirty) for dirty in all_users_raw_nested_tuples]
    print(all_users_clean_nested_tuples[0])

    return 

def make_splits(user_arr: np.ndarray) -> tuple:
    splits = None
    return splits

def main():
    # Data import
    default_raw_path = PurePath("../../data/clarkson2_files/")
    user_list = [user for user in os.listdir(default_raw_path) if user != ".DS_Store"]
    path = lambda user: PurePath(default_raw_path, PurePath(str(user)))
    read_paths = [path(user) for user in user_list]

    list_of_user_arrays = read_in_data(read_paths)

    # For each user, perform cross validation
    for i, user_arr in enumerate(list_of_user_arrays):

        # Determine splits
        splits = make_splits()

        # Generate arrays
        arrays_by_splits = [user_arr[split] for split in splits]

        # Start a simulation with each array
        # results = [simulate(array) for array in arrays_by_splits]

    pass

if __name__ == "__main__":
    main()
