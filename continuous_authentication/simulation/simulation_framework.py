import numpy as np
from pathlib import PurePath
import os
import csv
import sys
sys.path.append("../../")

from continuous_authentication.feature_extraction.parse_utils import *
from continuous_authentication.simulation.models import *

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

def digraphs_in_file(file):
    input_file = open(file,"r+")
    reader_file = csv.reader(input_file, delimiter = "\t")
    csv_ = list(reader_file)
    all_digraphs = [int(csv_[i][2]) for i in range(len(csv_))]
    sum_of_digraphs = sum(all_digraphs)

    return sum_of_digraphs
         
def process_train(train):
    user_profile_dict = {}
    for word in train:
        time, text, digraph_cnt, ngraph_str = word
        ngraph_vector = [int(ngraph) for ngraph in ngraph_str.lstrip("[").rstrip("]").split(", ")]

        if text in user_profile_dict:
            user_profile_dict[text]["occurence_count"] += 1
            user_profile_dict[text]["timing_vectors"].append([time, ngraph_vector])
        
        else:
            user_profile_dict[text] = {
                "occurence_count": 1,
                "timing_vectors": [[time, ngraph_vector]]
            }

    return user_profile_dict

def model_wrapper(user_profile, test_sample, model, threshold):
    word_profiles = {}
    for word, contents in user_profile.items():
        if contents["occurence_count"] >= threshold:
            # if a word hits the threshold, calculate the mean of all instances of that word and assign to word_profiles
            word_profiles[word] = np.array([vector[1] for vector in contents["timing_vectors"]]).mean(axis = 0)

def main(model, train_digraphs = 10000, test_digraphs = 1000, word_count_threshold = 3):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # Data import
    default_raw_path = PurePath("../../data/clarkson2_files/")
    default_timeseries_path = PurePath("../../data/user_time_series/")
    user_list = [user for user in os.listdir(default_raw_path) if user not in (".DS_Store", "379149")]
    path = lambda user: PurePath(default_timeseries_path, PurePath(f"user_{user}.csv"))
    read_paths = [path(user) for user in user_list]
    dt = np.dtype([('timestamp', np.uint32), ("text", np.unicode_, 16), ("digraphs", np.uint8), ("timing_vector", np.unicode_, 2048)])

    list_of_user_arrays = [np.genfromtxt(path, dtype = np.dtype(dt), delimiter = "\t") for path in read_paths]

    imposter_bank = []
    results_dict = {}
    # For each user, perform cross validation
    for i, user_arr in enumerate(list_of_user_arrays):
        if i != 37:
            continue
        num_digraphs_in_file = digraphs_in_file(read_paths[i])

        if num_digraphs_in_file <  train_digraphs + test_digraphs:
            continue 

        # Pull out training data
        train, test, remainder = train_test_remainder(user_arr, train_digraphs = 10000, test_digraphs = 1000)

        # Add partitioned remainder to imposter_bank to allow for testing random imposters
        partitioned_remainder = partition(remainder, test_digraphs = 1000)
        [imposter_bank.append(sublist) for sublist in partitioned_remainder]

        # Process training data into user model
        rng = np.random.default_rng()
        imposter_blocks = rng.choice(imposter_bank, 10, ).tolist() # 10 is number of imposter blocks to sample
        data = [train, test, *imposter_blocks]
        user_profile, genuine_sample, *imposter_samples = [process_train(datum) for datum in data]

        # Test genuine user
        genuine_results = model_wrapper(user_profile, test, model, word_count_threshold)
        imposter_results = [model_wrapper(user_profile, imposter_sample, model, word_count_threshold) for imposter_sample in imposter_samples]
    pass

def main_set_params():
    main(
        train_digraphs = 10000, 
        test_digraphs = 1000, 
        word_count_threshold = 2,
        model = Scaled_Manhattan,
    )

if __name__ == "__main__":
    main_set_params()
