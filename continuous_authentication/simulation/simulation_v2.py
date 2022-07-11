from typing import Iterable
import numpy as np
from pathlib import PurePath
import os
import csv
from json import dump
from time import perf_counter
import sys
sys.path.append(os.getcwd())
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

def model_wrapper(user_profile, test_sample, model, word_count_threshold, threshold, iter, genuine):

    # Take all words that occur more than threshold times as a subset of user_profile, prep for modelling
    word_profiles = set()
    for word, contents in user_profile.items():
        if contents["occurence_count"] >= word_count_threshold:
            # if a word hits the threshold, calculate the mean of all instances of that word and assign to word_profiles
            # word_profiles[word] = np.array([vector[1] for vector in contents["timing_vectors"]]).mean(axis = 0)
            word_profiles.add(word)
    counter = 0
    dissimilarity_vector = []
    word_lengths = []
    for word in test_sample:
        if word in word_profiles:
            train = np.array([vector[1] for vector in user_profile[word]["timing_vectors"]]) / 1000000
            # counter +=1
            for instance in test_sample[word]["timing_vectors"]:
                # maybe increment a counter here instead
                # counter +=1
                word_lengths.append(len(word))
                dissimilarity = model(train, instance[1])[0]
                dissimilarity_vector.append(dissimilarity)

    ## Logging
    # if iter == 0:
    #     with open(f"word_occurences.csv", 'a', encoding = "utf-8") as f:
    #             f.write(f"{'-' if genuine else ''}{counter}\n")

    if dissimilarity_vector:
        dissimilarity_array = np.array(dissimilarity_vector) / 1000000
        decisions = np.where(dissimilarity_array > threshold, 0, 1).tolist()

    else: # if no items are in both user profile and test sample, return None
        decisions = 0

    return decisions, word_lengths

def mean(x):
    try: 
        result = sum(x) / len(x)

    except ZeroDivisionError:
        raise("ZeroDivisionError: Empty intersect between train and test")

    return result

def write_to_csv(results, path):
    with open(path, "w") as f:
        dump(results, f, indent = 4)

    pass

def main(model, threshold_params,train_digraphs = 10000, test_digraphs = 1000, word_count_threshold = 3):
    pass

# def main_set_params():
#     start = perf_counter()
#     main(
#         train_digraphs = 10000, 
#         test_digraphs = 1000, 
#         word_count_threshold = 2,
#         model = Manhattan,
#         threshold_params = [0, 60, 5]
#     )
#     stop = perf_counter()
#     print(f"Total execution time: {stop - start}")

def simulation(
    input_folder: PurePath,
    distance_threshold_params: dict,
    occurence_threshold: int, 
    instance_threshold: int,
    distance_metric: function,
    train_word_count: int,
    test_word_count: int,
):
    # Read in each user's time series stream of typed words

    # Remove all time series' with fewer than train_word_count + test_word_count words

    # 

def single_main():
    results = simulation(
        input_folder = PurePath("data/user_time_series/"),
        distance_threshold_params = {"start": 0, "stop": 10, "step": 3},
        occurence_threshold = 3, 
        instance_threshold = 5,
        distance_metric = Manhattan,
        train_word_count = 1000,
        test_word_count = 50
    )

if __name__ == "__main__":
    # main_set_params()
    single_main()