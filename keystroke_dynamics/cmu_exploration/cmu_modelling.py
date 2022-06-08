import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import *

import sys
sys.path.append("keystroke_dynamics")
from cmu_exploration.models import *


def process_input_data(df): 
    subjects = df.subject.unique()
    dfs_list_by_subject = [df[df.subject == subject] for subject in subjects]
    nested_list_by_subject = [df.values.tolist() for df in dfs_list_by_subject]
    matrix_by_subject = np.array(nested_list_by_subject)
    data = np.copy(matrix_by_subject)
    X = data[:, :, 3:]
    y = data[:, :, 0]

    return X, y


def cross_validate(X, thresholds, rng, model):
    # Manual cross-validation because pipelines aren't suited for the statistical methods
    user_scores = {threshold: [] for threshold in thresholds}

    for user_num, user in enumerate(X):
        # if user_num == 0: 
        # create object to scale data
        scaler = StandardScaler()
        scaled_user = scaler.fit_transform(user)

        # select 200 of 400 samples to use as train, other 200 go to testing for that user
        random_user_indices = np.arange(400)
        rng.shuffle(random_user_indices)
        train_indices = random_user_indices[:200]
        test_user_indices = random_user_indices[200:]
        train = scaled_user[train_indices]
        test_user = scaled_user[test_user_indices]

        # print("Train")
        # print(train[:10])

        # select 5 of 400 samples from each other user to use as imposter testing
        other_user_nums = np.concatenate((np.arange(51)[:user_num], np.arange(51)[user_num+1:]))
        imposter_indices = rng.choice(400, 5)
        test_imposters = X[other_user_nums][:, imposter_indices].reshape((250, 31)).astype("float64")
        test_imposters = scaler.transform(test_imposters)

        # compare both test_user and test_imposters against train using some model at various thresholds
        for i, threshold in enumerate(thresholds):
            user_pred_labels = model(train, test_user, threshold)
            imposter_pred_labels = model(train, test_imposters, threshold)

            user_pred_tpr = np.sum(user_pred_labels) / user_pred_labels.size
            imposter_pred_fpr = np.sum(imposter_pred_labels) / imposter_pred_labels.size

            user_scores[threshold].append((user_pred_tpr, imposter_pred_fpr))

    return user_scores


def process_output_data(scores):
    user_scores_arr_dict = {key: np.array(value) for key, value in scores.items()}
    user_scores_avg_dict = {key: np.mean(value, axis = 0).tolist() for key, value in user_scores_arr_dict.items()}
    tpr_fpr_scores = [val for val in user_scores_avg_dict.values()]
    tpr_scores = [item[0] for item in tpr_fpr_scores]
    fpr_scores = [item[1] for item in tpr_fpr_scores]

    return tpr_scores, fpr_scores


def plot_ROC_curve(tpr, fpr, thresholds, performance, model_name, output_folder):
    fig, ax = plt.subplots()
    ax.fill_between(fpr, tpr)
    sns.scatterplot(x = fpr, y = tpr, ax = ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    v, h = .1, .1
    ax.set_xlim(0-h, 1+h)
    ax.set_ylim(0-v, 1+v)

    # Loop through the data points 
    for i, threshold in enumerate (thresholds):
        plt.text(fpr[i], tpr[i], threshold)

    vals = [i for i in np.arange(0, 1, 0.01)]
    xp = [i for i in np.arange(1, 0, -0.01)]
    sns.lineplot(x = vals, y = vals, ax = ax, color = "red")
    sns.lineplot(x = vals, y = xp, ax = ax, color = "green")
    fig.suptitle(f"ROC Curve for {model_name}")
    fig.set_size_inches(10, 7)
    start = .2
    gap = .2
    height = 1.05
    plt.text(start, height, f"EER: {round(performance['EER'], 3) * 100}%")
    plt.text(start + gap, height, f"Threshold: {round(performance['Threshold'], 3)}")
    plt.text(start + 2*gap + 0.05, height, f"AUC: {round(performance['AUC'], 3)}")

    plt.savefig(f"{output_folder}{model_name}", dpi = 400)

    pass

def format_plots():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 26
    CHONK_SIZE = 32
    font = {'family' : 'DIN Condensed',
            'weight' : 'bold',
            'size'   : SMALL_SIZE}
    plt.rc('font', **font)
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:white")
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black") #  powder blue

    pass

def calc_model_performance(tpr, fpr, thresholds):
    performance_dict = {}
    
    # calculating auc
    performance_dict["AUC"] = np.trapz(y = tpr, x = fpr)
    
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


def main(data_in, seed, t_start, t_stop, t_step, model, graph_output_folder): 
    # read in data
    df = pd.read_csv(data_in, sep = ",", header = 0)
    # establish random number generator
    rng = np.random.default_rng(seed)
    format_plots()
    X, y = process_input_data(df)
    threshold_lst = [round(i, 2) for i in np.arange(t_start, t_stop, t_step)]
    tpr_fpr_dict = cross_validate(X, threshold_lst, rng, model)
    tpr, fpr = process_output_data(tpr_fpr_dict)
    performance = calc_model_performance(tpr, fpr, threshold_lst)
    print(performance)
    plot_ROC_curve(tpr, fpr, threshold_lst, performance, model.__name__, graph_output_folder)



def main_wrapper():
    kwargs = {
        "data_in": "keystroke_dynamics/data/cmu_data.txt",
        "seed": 8675309,
        "t_start": 0,
        "t_stop": 3, 
        "t_step": 0.05, 
        "model": z_score,  
        "graph_output_folder": "keystroke_dynamics/cmu_exploration/graphics/",
    }

    main(**kwargs)

if __name__ == "__main__":
    main_wrapper()

