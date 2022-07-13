from pathlib import PurePath
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from time import perf_counter as pfc
import sys
sys.path.append(os.getcwd())
from parse_utils import file_to_nested_tuples

def ticks_to_datetime(binary_time: int) -> datetime:
    binary_time_string = f"{binary_time:064b}"
    time_microseconds = int(binary_time_string[2:], 2) / 10
    time_difference = timedelta(microseconds=time_microseconds)
    return datetime(1, 1, 1) + time_difference

def ticks_to_unix(binary_time):
    """ 
    Converts ticks to unix timestamp in milliseconds (divide by 1000 for seconds) 
    """
    dt = ticks_to_datetime(binary_time)
    return dt.timestamp()

def main():
    read_path = PurePath("data/clarkson2_files/")
    write_path = PurePath("data/clarkson2_files_unix/")
    files = os.listdir(read_path)
    for i, file in enumerate(files):
        print(f"{i}: {file}")
        df = pd.read_csv(PurePath(read_path, PurePath(file)), sep = "\t", header = None)
        df.iloc[:, 0] = df.iloc[:, 0].apply(ticks_to_unix)
        df.to_csv(PurePath(write_path, PurePath(file)), sep = "\t", header = None, index = False)
        # print(df)
        # contents = file_to_nested_tuples(PurePath(read_path, PurePath(file)))
        # fixed_time_contents = tuple()

if __name__ == "__main__":
    main()