import numpy as np

def read_in_data(path_lst: list) -> tuple(np.ndarray,np.ndarray,):
    list_of_user_arrays = []
    return list_of_user_arrays

def make_splits(user_arr: np.ndarray, interval: int = 30) -> tuple(np.ndarray,np.ndarray,):
    splits = None
    return splits

def main():
    # Data handling
    list_of_user_arrays = read_in_data(None)

    # For each user, perform cross validation
    for i, user_arr in enumerate(list_of_user_arrays):

        # Determine splits
        splits = make_splits()

        # Generate arrays
        arrays_by_splits = [user_arr[split] for split in splits]

        # Start a simulation with each array

    pass

if __name__ == "__main__":
    main()
    user_arr = None