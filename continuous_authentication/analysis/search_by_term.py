from pathlib import PurePath
import os

def main():
    default_time_series_path = PurePath("../../data/user_time_series/")
    default_raw_path = PurePath("../../data/clarkson2_files/")
    user_list = [user for user in os.listdir(default_raw_path) if user != ".DS_Store"]
    user_ts_path = PurePath(default_time_series_path, PurePath(f"user_{user_list[0]}.csv"))

    # for user in user_list: 
    #     user_ts_path = PurePath(default_time_series_path, PurePath(f"user_{user}.csv"))

    for user in user_list:
        path = PurePath(default_time_series_path, PurePath(f"user_{user}.csv"))
        with open(path, "r") as f:
            split_lines = [line.split() for line in f.readlines()]
            data = [(int(line[0]), line[1]) for line in split_lines]
            long_string = "."

if __name__ == "__main__":
    main()