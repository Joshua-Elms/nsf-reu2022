from typing import Callable
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

def simulation(
    input_folder: PurePath,
    output_folder: PurePath,
    distance_threshold_params: dict,
    occurence_threshold: int, 
    instance_threshold: int,
    distance_metric: Callable,
    train_word_count: int,
    test_word_count: int,
    decisions: int
):
    # Initialize list of thresholds to run model with

    # Get run number
    iteration = get_iter(output_folder)

    print(iteration)

    # Read in each user's time series stream of typed words

    # Remove all time series' with fewer than train_word_count + test_word_count words

    # 

    pass

def single_main():
    results = simulation(
        input_folder = PurePath("data/user_time_series/"),
        output_folder = PurePath("continuous_authentication/simulation/results/"),
        distance_threshold_params = {"start": 0, "stop": 10, "step": 3},
        occurence_threshold = 3, 
        instance_threshold = 5,
        distance_metric = Manhattan,
        train_word_count = 1000,
        test_word_count = 50,
        decisions = 5
    )

if __name__ == "__main__":
    # main_set_params()
    single_main()