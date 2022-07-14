from typing import Callable
import numpy as np
from pathlib import PurePath
import os
import re
from sklearn.metrics import roc_curve, roc_auc_score
import pyaml
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from time import perf_counter
import sys

sys.path.append(os.getcwd())
from continuous_authentication.feature_extraction.parse_utils import *
from continuous_authentication.simulation.models import *


def get_iter(counter_folder, increment):

    iter_num_path = PurePath(counter_folder, PurePath("iter_num.txt"))
    try:
        with open(iter_num_path, "r") as f_r:
            iter_str = f_r.readline().strip()

            try:
                iter = int(iter_str)

            except ValueError:
                iter = None
                print(
                    f"No valid iteration counter detected, file: {iter_num_path} added and set to 1\n"
                )

            if isinstance(iter, int):
                next_iter = iter + 1

            else:
                next_iter = 1

        with open(iter_num_path, "w") as f_w:
            f_w.write(str(next_iter))

    except FileNotFoundError:
        with open(iter_num_path, "w") as f_w:
            f_w.write("2")
        iter = 1
    return iter


def read_data(data_folder):
    # Generate a list of all csv's in data folder
    files = os.listdir(data_folder)
    get_ext = lambda f: os.path.splitext(f)[1]
    csv_files = [file for file in files if get_ext(file) == ".csv"]

    # Define datatype for file
    dt = np.dtype(
        [
            ("timestamp", np.float64),
            ("text", np.unicode_, 16),
            ("digraphs", np.uint8),
            ("timing_vector", np.unicode_, 2048),
        ]
    )

    # Create list of all user csv's as ndarrays
    csv_path = lambda folder, file: PurePath(folder, PurePath(file))
    all_arrays = [
        np.genfromtxt(csv_path(data_folder, path), dtype=np.dtype(dt), delimiter="\t")
        for path in csv_files
    ]

    numbers = lambda str: re.findall("[0-9]+", str)[0]
    users = [numbers(file) for file in files]

    return all_arrays, users


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
        ngraph_vector = [
            float(ngraph) for ngraph in ngraph_str.lstrip("[").rstrip("]").split(", ")
        ]

        if text in user_profile_dict:
            user_profile_dict[text]["occurence_count"] += 1
            user_profile_dict[text]["timing_vectors"].append([time, ngraph_vector])

        else:
            user_profile_dict[text] = {
                "occurence_count": 1,
                "timing_vectors": [[time, ngraph_vector]],
            }

    return user_profile_dict


def get_words_for_decision(profile, test, o_threshold, i_threshold, start):
    # Filter out any words in profile w/ fewer than o_threshold occurences
    valid_profile = {
        word: content
        for word, content in profile.items()
        if content["occurence_count"] >= o_threshold
    }

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
        ngraph_vector = [
            float(ngraph) for ngraph in ngraph_str.lstrip("[").rstrip("]").split(", ")
        ]

        # Compares instances, not unique words
        if word in valid_profile:
            instance_cnt += 1
            shared_words.append((word, ngraph_vector))

    return shared_words, timestamp, furthest_idx_pos_reached


def make_decision(profile, test, dist, fusion, normalize):
    # Pass over each word in test and use dist to compare it to the profile, then add distance to distances list
    distances = []
    weights = []
    for instance in test:
        word, test_ngraph_vector = instance
        word_in_profile = profile[word]

        # pull out all timing vectors for that word in the training profile
        train_ngraph_nest_lst = [
            word_in_profile["timing_vectors"][i][1]
            for i in range(word_in_profile["occurence_count"])
        ]
        train_graph_matrix = np.array(train_ngraph_nest_lst)

        # initialize weights according to fusion parameter
        if fusion == "equal":
            weights.append(1)

        elif fusion == "proportional_to_length":
            weights.append(len(word))

        elif fusion == "inv_proportional_to_stdev":
            std = np.std(train_graph_matrix, axis=0)
            mean_std = std.mean()
            inv_mean_std = 1 / mean_std
            weights.append(inv_mean_std)

        # calculate the distance from mean of training vectors to test vector
        distance = dist(X=train_graph_matrix, y=test_ngraph_vector)
        distances.append(distance)

    # multiply votes by their respective weights and then sum
    weights_matrix = np.array(weights)
    weights_x_dists = weights_matrix * distances
    fused_score = np.sum(weights_x_dists)

    return fused_score


def perform_decision_cycle(
    distance_metric,
    occurence_threshold,
    instance_threshold,
    num_decisions,
    normalize_data,
    accumulator,
    id,
    gen_or_imp,
    user_profile,
    test_array,
    weighting,
):
    # Compare genuine array to user_profile until specified # of decisions made
    for decision_num in range(num_decisions):
        # First iter needs to init last_idx_pos as starting position for get_words
        if decision_num == 0:
            last_idx_pos = 0

        # No decision occurs here, just grabbing the first i_threshold words that are shared by profile

        words_for_decision, decision_timestamp, last_idx_pos = get_words_for_decision(
            profile=user_profile,
            test=test_array,
            o_threshold=occurence_threshold,
            i_threshold=instance_threshold,
            start=last_idx_pos,
        )

        # Distance calc for each timing vector in words_for_decision, compared against each of decision_threshold
        score = make_decision(
            profile=user_profile,
            test=words_for_decision,
            dist=distance_metric,
            fusion=weighting,
            normalize=normalize_data,
        )

        # Add results to accumulator
        accumulator[f"user_{id}"][f"{gen_or_imp}_scores"].append(score)

        if decision_num > 0:
            interval = decision_timestamp - last_decision_timestamp
            accumulator[f"user_{id}"][f"{gen_or_imp}_intervals"].append(interval)

        last_decision_timestamp = decision_timestamp


def simulation(
    input_folder: PurePath,
    distance_metric: Callable,
    occurence_threshold: int,
    instance_threshold: int,
    train_word_count: int,
    num_imposters: int,
    num_imposter_decisions: int,
    num_genuine_decisions: int,
    word_count_scale_factor: int,
    user_cnt: int,
    normalize_data: bool,
    weighting: str,
):
    # Mask warning from reading empty file
    np.seterr(all="ignore")

    start_read = perf_counter()
    # Read in each user's time series stream of typed words
    all_user_timeseries, user_ids = read_data(input_folder)
    end_read = perf_counter()

    print(f"Time to read in data: {round(end_read - start_read, 2)} seconds")

    # Remove all time series' with fewer than minimum words
    minimum_words = train_word_count + (
        num_genuine_decisions * instance_threshold * word_count_scale_factor
    )
    valid_arrays = [
        (i, arr) for i, arr in enumerate(all_user_timeseries) if get_array_length(arr) >= minimum_words
    ]

    # Only take first user_cnt arrays, raise error if parameter is invalid
    try:
        desired_arrays = valid_arrays[:user_cnt]
        num_users = len(desired_arrays)

    except IndexError:
        raise (
            ValueError(
                f"Only {len(valid_arrays)} users present, you passed user_cnt = {user_cnt}"
            )
        )

    # Split each array into train and test
    train_arrays = []
    test_arrays = []
    for i, array in desired_arrays:
        train_arrays.append(array[:train_word_count])
        test_arrays.append(array[train_word_count:])

    # Generate user_profiles from training data, used for comparison against test
    user_profiles = [process_train(array) for array in train_arrays]

    start_process = perf_counter()
    # Main Loop will iterate over each user to find scores and decision intervals
    result_accumulator = {}
    for i, arr in desired_arrays:
        result_accumulator[f"user_{user_ids[i]}"] = {
            "genuine_scores": [],
            "imposter_scores": [],
            "genuine_intervals": [],
            "imposter_intervals": []
        }

    for idx in range(num_users):

        # get this user's data
        user_id = user_ids[desired_arrays[idx][0]]
        user_profile = user_profiles[idx]
        genuine_array = test_arrays[idx]

        # any array other than the current is an imposter
        rng = np.random.default_rng()
        non_user_arrays = np.array(test_arrays[:idx] + test_arrays[idx + 1 :], dtype=object)
        imposter_arrays = rng.choice(
            non_user_arrays,
            size=min(num_imposters, num_users - 1),
            replace=False
        )

        ### Calc TPR ###
        perform_decision_cycle(
            distance_metric=distance_metric,
            occurence_threshold=occurence_threshold,
            instance_threshold=instance_threshold,
            num_decisions=num_genuine_decisions,
            normalize_data=normalize_data,
            accumulator=result_accumulator,
            id=user_id,
            gen_or_imp="genuine",
            user_profile=user_profile,
            test_array=genuine_array,
            weighting=weighting,
        )

        ### Calc FPR ###
        for imposter_array in imposter_arrays:
            perform_decision_cycle(
                distance_metric=distance_metric,
                occurence_threshold=occurence_threshold,
                instance_threshold=instance_threshold,
                num_decisions=num_imposter_decisions,
                normalize_data=normalize_data,
                accumulator=result_accumulator,
                id=user_id,
                gen_or_imp="imposter",
                user_profile=user_profile,
                test_array=imposter_array,
                weighting=weighting,
            )

    end_process = perf_counter()
    print(f"Time to process data: {round(end_process - start_process, 2)} seconds")

    return result_accumulator


def calc_model_performance(tpr, fpr, thresholds):
    performance_dict = {}

    # calculating auc
    performance_dict["AUC"] = np.trapz(y=tpr, x=fpr)

    # calculating eer
    tnr = 1 - np.array(tpr)
    scores = np.array((tnr, fpr)).T
    diffs = np.absolute(scores[:, 0] - scores[:, 1])
    min_index = np.argmin(diffs)
    lowest_threshold = thresholds[min_index]
    eer = (tnr[min_index] + fpr[min_index]) / 2
    performance_dict["EER"] = eer
    performance_dict["Threshold"] = lowest_threshold

    return performance_dict


def plot_ROC_curve(tpr, fpr, thresholds, performance, run_num, output_folder):
    fig, ax = plt.subplots()
    ax.fill_between(fpr, tpr)
    sns.scatterplot(x=fpr, y=tpr, ax=ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    v, h = 0.1, 0.1
    ax.set_xlim(0 - h, 1 + h)
    ax.set_ylim(0 - v, 1 + v)

    # Loop through the data points
    for i, threshold in enumerate(thresholds):
        plt.text(fpr[i], tpr[i], threshold)

    vals = [i for i in np.arange(0, 1, 0.01)]
    xp = [i for i in np.arange(1, 0, -0.01)]
    sns.lineplot(x=vals, y=vals, ax=ax, color="red")
    sns.lineplot(x=vals, y=xp, ax=ax, color="green")
    fig.suptitle(f"ROC Curve: Simulation {run_num}")
    fig.set_size_inches(10, 7)
    start = 0.2
    gap = 0.2
    height = 1.05
    plt.text(start, height, f"EER: {round(performance['EER'] * 100, 1)}%")
    plt.text(start + gap, height, f"Threshold: {round(performance['Threshold'], 3)}")
    plt.text(start + 2 * gap + 0.05, height, f"AUC: {round(performance['AUC'], 3)}")

    out = PurePath(output_folder, PurePath(f"roc_curve_{run_num}.png"))
    plt.savefig(out, dpi=400)


def dump_to_yaml(path, object):
    with open(path, "w") as f_log:
        dump = pyaml.dump(object)
        f_log.write(dump)


def format_plots():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 26
    CHONK_SIZE = 32
    font = {"family": "DIN Condensed", "weight": "bold", "size": SMALL_SIZE}
    plt.rc("font", **font)
    plt.rc("axes", titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:white")
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc(
        "figure", titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black"
    )  #  powder blue


def postprocessing(
    simulation_params,
    tpr_agg,
    fpr_agg,
    genuine_intervals,
    imposter_intervals,
    directory,
):

    # allocating a folder for each run so that metadata and model output can be contained nicely
    iteration = get_iter(directory, increment=True)
    folder_name = f"simulation_{iteration}"
    fullpath = PurePath(directory, PurePath(folder_name))
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    # if this step is skipped, yaml dump tries and fails to write actual distance metric to output
    metadata = deepcopy(simulation_params)
    metadata["distance_metric"] = metadata["distance_metric"].__name__

    # log metadata to yaml file so that simulation conditions can be replicated and reported
    metadata_path = PurePath(fullpath, PurePath(f"params_{iteration}.yaml"))
    dump_to_yaml(metadata_path, metadata)

    # generate thresholds the same as in simulation
    dist_thresholds = np.round(
        np.arange(**simulation_params["distance_threshold_params"]), 2
    )

    # both of these will be lists, better for graphing and writing out
    tpr_by_thresh, fpr_by_thresh = calc_tpr_fpr(tpr_agg), calc_tpr_fpr(fpr_agg)

    # calculations for EER and optimal threshold are debatable, but I used the most reliable one from stack overflow
    model_perf = calc_model_performance(
        tpr=tpr_by_thresh, fpr=fpr_by_thresh, thresholds=dist_thresholds
    )

    # log model performance for meta-analysis
    perf_path = PurePath(fullpath, PurePath(f"performance_{iteration}.yaml"))
    dump_to_yaml(perf_path, model_perf)

    # Set formatting for plots
    format_plots()

    # Plot ROC Curve
    plot_ROC_curve(
        tpr=tpr_by_thresh,
        fpr=fpr_by_thresh,
        thresholds=dist_thresholds,
        performance=model_perf,
        run_num=iteration,
        output_folder=fullpath,
    )

    print(f"Saved output of postprocessing for simulation {iteration}")


def single_main():
    ts_data = PurePath("data/user_time_series/")
    results_folder = PurePath("continuous_authentication/simulation/results/")
    simulation_parameters = {
        "distance_metric": Euclidean,
        "occurence_threshold": 3,
        "instance_threshold": 5,
        "train_word_count": 1000,
        "num_imposters": 10,
        "num_imposter_decisions": 3,
        "num_genuine_decisions": 30,
        "word_count_scale_factor": 30,
        "user_cnt": -1,  # -1 yields all users
        "normalize_data": False,
        "weighting": "proportional_to_length",
    }

    results = simulation(input_folder=ts_data, **simulation_parameters)
    print(results)
    # postprocessing(
    #     simulation_params=simulation_parameters,
    #     tpr_agg=results[0],
    #     fpr_agg=results[1],
    #     genuine_intervals=results[2],
    #     imposter_intervals=results[3],
    #     directory=results_folder,
    # )


if __name__ == "__main__":
    single_main()
