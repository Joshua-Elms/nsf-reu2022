import numpy as np
from pathlib import PurePath
import os
import csv
import sys
sys.path.append("../../")

from continuous_authentication.feature_extraction.parse_utils import *

def train_test_remainder(user_arr: np.ndarray, train_digraphs: int, test_digraphs: int) -> tuple:
    eof = len(user_arr) 

    train_limit = 0
    digraph_cnt = 0
    for i, val in enumerate(user_arr):
        digraphs = int(val[2])
        if digraph_cnt + digraphs <= test_digraphs:
            digraph_cnt += digraphs

        else: 
            train_limit = i
            digraph_cnt = 0
            break

    test_limit = 0
    for i, val in enumerate(user_arr):
        if i > train_limit:
            digraphs = int(val[2])
            if digraph_cnt + digraphs <= test_digraphs:
                digraph_cnt += digraphs

            else: 
                test_limit = i
                break

    train = user_arr[ : train_limit]
    test = user_arr[train_limit : test_limit]
    remainder = user_arr[test_limit : ]

    return train, test, remainder


def digraphs_in_file(file):
    input_file = open(file,"r+")
    reader_file = csv.reader(input_file, delimiter = "\t")
    csv_ = list(reader_file)
    all_digraphs = [int(csv_[i][2]) for i in range(len(csv_))]
    sum_of_digraphs = sum(all_digraphs)

    return sum_of_digraphs

def partition(data, test_digraphs):
    chunk_lst = []
    digraph_cnt = 0
    previous_lim = 0
    for i, val in enumerate(data):
        digraphs = int(val[2])
        if digraph_cnt + digraphs <= test_digraphs:
            digraph_cnt += digraphs

        else: 
            chunk_lst.append(data[previous_lim : i])
            previous_lim = i
            digraph_cnt = 0

    return chunk_lst
            
def process_train(train, occurence_threshold = 3):
    pass

def main(train_digraphs = 10000, test_digraphs = 1000):


    # Data import
    default_raw_path = PurePath("../../data/clarkson2_files/")
    default_timeseries_path = PurePath("../../data/user_time_series/")
    user_list = [user for user in os.listdir(default_raw_path) if user != ".DS_Store"]
    path = lambda user: PurePath(default_timeseries_path, PurePath(f"user_{user}.csv"))
    read_paths = [path(user) for user in user_list]

    list_of_user_arrays = [np.genfromtxt(path, dtype = np.dtype(object), delimiter = "\t") for path in read_paths]

    imposter_bank = []
    results_dict = {}
    # For each user, perform cross validation
    for i, user_arr in enumerate(list_of_user_arrays):
        if i != 37:
            break
        num_digraphs_in_file = digraphs_in_file(read_paths[i])

        if num_digraphs_in_file <  train_digraphs + test_digraphs:
            continue 

        # Pull out training data
        train, test, remainder = train_test_remainder(user_arr, train_digraphs = 10000, test_digraphs = 1000)

        # Add partitioned remainder to imposter_bank to allow for testing random imposters
        partitioned_remainder = partition(remainder, test_digraphs = 1000)
        [imposter_bank.append(sublist) for sublist in partitioned_remainder]

        # Process training data into user model
        user_profile = process_train(train, occurence_threshold = 3)

    pass

if __name__ == "__main__":
    main(train_digraphs = 10000, test_digraphs = 1000)
