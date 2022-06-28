import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import PurePath
from dask import dataframe
import os

def read_in_data(path_lst): 
    frame_datas = [dataframe.read_csv(path, sep = "\t") for path in path_lst]

    return frame_datas

def main():
    # Data import
    default_raw_path = PurePath("../../data/clarkson2_files/")
    default_timeseries_path = PurePath("../../data/user_time_series/")
    user_list = [user for user in os.listdir(default_raw_path) if user != ".DS_Store"]
    path = lambda user: PurePath(default_timeseries_path, PurePath(f"user_{user}.csv"))
    read_paths = [path(user) for user in user_list[:3]]

    list_of_user_arrays = read_in_data(read_paths)
    print(list_of_user_arrays)


if __name__ == "__main__":
    main()