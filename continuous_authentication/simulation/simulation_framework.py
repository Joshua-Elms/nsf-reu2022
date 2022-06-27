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
    get_actual_keystrokes = lambda indices, keystrokes: [keystrokes[i] if i >= 0  else (None, None, "-") for i in indices]
    cleaned_keystrokes = [get_actual_keystrokes(all_users_clean_nested_tuples[i], all_users_raw_nested_tuples[i]) for i in range(len(all_users_raw_nested_tuples))]

    return cleaned_keystrokes

def determine_folds(user_arr: np.ndarray, k_folds: int = 5, p_left_out: int = 2) -> tuple:
    rng = np.random.default_rng()
    

    return #folds

def main():
    # Data import
    default_raw_path = PurePath("../../data/clarkson2_files/")
    user_list = [user for user in os.listdir(default_raw_path) if user != ".DS_Store"]
    path = lambda user: PurePath(default_raw_path, PurePath(str(user)))
    read_paths = [path(user) for user in user_list[:3]]

    list_of_user_arrays = read_in_data(read_paths)

    # For each user, perform cross validation
    for i, user_arr in enumerate(list_of_user_arrays):
        print(f"Successfully did fuck all for user: {i}")
        # Determine splits
        folds = determine_folds(user_arr)
        print(folds)

        # Generate arrays
        # arrays_by_folds = [user_arr[fold] for fold in folds]

        # Start a simulation with each array
        # results = [simulate(array) for array in arrays_by_folds]

    pass

if __name__ == "__main__":
    main()
