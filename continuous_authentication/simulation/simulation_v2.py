from multiprocessing.sharedctypes import Value
from typing import Callable
import numpy as np
from pathlib import PurePath
import os
from time import perf_counter
import sys
sys.path.append(os.getcwd())
from continuous_authentication.feature_extraction.parse_utils import *
from continuous_authentication.simulation.models import *

def get_iter(counter_folder):

    iter_num_path = PurePath(counter_folder, PurePath("iter_num.txt"))
    with open(iter_num_path, "r") as f_r:
        iter_str = f_r.readline().strip()

        try: 
            iter = int(iter_str)

        except ValueError:
            iter = None
            print(f"No valid iteration counter detected, file: {iter_num_path} added and set to 1")

        if isinstance(iter, int):
            next_iter = iter + 1
        
        else:
            next_iter = 1

    with open(iter_num_path, "w") as f_w:
        f_w.write(str(next_iter))
        
    return iter        

def read_data(data_folder):
    # Generate a list of all csv's in data folder
    files = os.listdir(data_folder)
    get_ext = lambda f: os.path.splitext(f)[1]
    csv_files = [file for file in files if get_ext(file) == ".csv"]

    # Define datatype for file
    dt = np.dtype([
        ('timestamp', np.uint32),
        ("text", np.unicode_, 16),
        ("digraphs", np.uint8),
        ("timing_vector", np.unicode_, 2048)
        ])

    # Create list of all user csv's as ndarrays
    csv_path = lambda folder, file: PurePath(folder, PurePath(file))
    all_arrays = [np.genfromtxt(csv_path(data_folder, path), dtype = np.dtype(dt), delimiter = "\t") for path in csv_files]

    return all_arrays

def get_array_length(array):
    try:
        length = array.shape[0]

    except Exception:
        length = 0

    return length

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

def get_words_for_decision(
    profile, 
    test, 
    o_threshold, 
    i_threshold, 
    start
    ):
    # Filter out any words in profile w/ fewer than o_threshold occurences
    valid_profile = {word: content for word, content in profile.items() if content["occurence_count"] >= o_threshold}

    # iterate through test data from designated start to finish until instance counter reaches threshold
    instance_cnt = 0
    shared_words = []
    furthest_idx_pos_reached = 0
    for i in range(start, len(test)):

        # End loop if instance counter hits i_threshold
        if instance_cnt == i_threshold:
            furthest_idx_pos_reached = i
            break

        # Parse line into components
        timestamp, word, digraph_cnt, ngraph_str = test[i]
        ngraph_vector = [int(ngraph) for ngraph in ngraph_str.lstrip("[").rstrip("]").split(", ")]

        # Compares instances, not unique words
        if word in valid_profile:
            instance_cnt += 1


        # Populate shared_word list with tuples containing the word string and timing vector (lst)
        shared_words.append((word, ngraph_vector))

    print(f"Decision timestamp: {timestamp}")

    return shared_words, timestamp, furthest_idx_pos_reached

def simulation(
    input_folder: PurePath,
    output_folder: PurePath,
    distance_metric: Callable,
    distance_threshold_params: dict,
    occurence_threshold: int, 
    instance_threshold: int,
    train_word_count: int,
    num_imposters: int, 
    imposter_decisions: int,
    genuine_decisions: int, 
    word_count_scale_factor: int,
    user_cnt: int
):
    # Mask warning from reading empty file
    np.seterr(all="ignore")

    # Initialize list of thresholds to run model with
    distance_thresholds = np.round(np.arange(**distance_threshold_params), 2)

    # Get run number for logging results
    iteration = get_iter(output_folder)

    # Read in each user's time series stream of typed words
    all_user_timeseries = read_data(input_folder)

    # Remove all time series' with fewer than minimum words
    minimum_words = train_word_count + (genuine_decisions * instance_threshold * word_count_scale_factor)
    valid_arrays = [arr for arr in all_user_timeseries if get_array_length(arr) >= minimum_words]
    del all_user_timeseries

    try:
        desired_arrays = valid_arrays[:user_cnt]
        num_users = len(valid_arrays)
        del valid_arrays

    except IndexError:
        raise(ValueError(f"Only {len(valid_arrays)} users present, you passed user_cnt = {user_cnt}"))


    # Split each array into train and test
    train_arrays = []
    test_arrays = []
    for array in desired_arrays:
        train_arrays.append(array[:train_word_count])
        test_arrays.append(array[train_word_count:])

    # Generate user_profiles from training data, used for comparison against test
    user_profiles = [process_train(array) for array in train_arrays]

    # Main Loop will iterate over each user to find TPR, FPR, and decision intervals
    result_dict = {}
    for idx in range(num_users):

        # get this user's data
        user_profile = user_profiles[idx]
        genuine_array = test_arrays[idx]

        # any array other than the current is an imposter
        imposter_arrays = test_arrays[:idx] + test_arrays[idx+1:]

        # Compare genuine array to user_profile until specified # of decisions made
        for decision_num in range(genuine_decisions):
            if decision_num == 0:
                last_idx_pos = 0
            words_for_decision, decision_timestamp, last_idx_pos = get_words_for_decision(profile = user_profile, 
                                                        test = genuine_array, 
                                                        o_threshold = occurence_threshold, 
                                                        i_threshold = instance_threshold,
                                                        start = last_idx_pos)

        # Compare imposter arrays to user_profile until specified # of decisions made per
        


def single_main():
    results = simulation(
        input_folder = PurePath("data/user_time_series/"),
        output_folder = PurePath("continuous_authentication/simulation/results/"),
        distance_metric = Manhattan,
        distance_threshold_params = {"start": 0, "stop": 10, "step": 3},
        occurence_threshold = 3, 
        instance_threshold = 5,
        train_word_count = 1000,
        num_imposters = 10,
        imposter_decisions = 2,
        genuine_decisions = 1,
        word_count_scale_factor = 50,
        user_cnt = -1
    )

if __name__ == "__main__":
    # main_set_params()
    single_main()