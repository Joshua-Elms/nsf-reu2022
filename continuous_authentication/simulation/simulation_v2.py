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
    genuine_decisions: int
):
    # Initialize list of thresholds to run model with
    distance_thresholds = np.arange(**distance_threshold_params)

    # Get run number
    iteration = get_iter(output_folder)

    # Read in each user's time series stream of typed words
    all_user_timeseries = read_data(input_folder)

    # Remove all time series' with fewer than train_word_count + test_word_count words
    minimum_words = train_word_count + 

    valid_arrays = [arr for arr in all_user_timeseries if get_array_length(arr) >= minimum_words]

    print(valid_arrays)

    # 

    pass

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
        genuine_decisions = 20
    )

if __name__ == "__main__":
    # main_set_params()
    single_main()