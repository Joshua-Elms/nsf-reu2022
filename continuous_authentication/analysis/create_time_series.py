from pathlib import PurePath
import os
from json import load
from datetime import datetime, timedelta
from time import mktime

def word_conditions(content):
    word, *_ = content
    is_word = True
    length = len(word)
    lower_incl = 2
    upper_incl = 15
    unique_letters = set(word)
    unique_total_min_ratio = .30

    # word inside length limits
    if length < lower_incl or length > upper_incl:
        is_word = False

    # ratio of unique:total letters must be higher than ratio to be considered real word
    elif (len(unique_letters) / length) < unique_total_min_ratio:
        is_word = False

    return is_word

def create_sorted_filtered_list(user_dict):
    tmp_lst = []
    for word in user_dict:
        timing_vectors = user_dict[word]["timing_vectors"]

        for instance in timing_vectors:
            timestamp = instance[0]
            tmp_lst.append((word, timestamp, instance[1]))

    filtered_tmp = tuple(filter(word_conditions, tmp_lst))
    sorted_lst = sorted(filtered_tmp, key = lambda x: x[1], reverse = False)

    return sorted_lst

def main():
    default_json_path = PurePath("data/user_json_files/")
    default_time_series_path = PurePath("data/user_time_series/")
    default_raw_path = PurePath("data/clarkson2_files/")
    user_list = [user for user in os.listdir(default_raw_path) if user != ".DS_Store"]

    for user in user_list: 
        user_json_path = PurePath(default_json_path, PurePath(f"user_{user}.json"))
        user_ts_path = PurePath(default_time_series_path, PurePath(f"user_{user}.csv"))

        with open(user_json_path, "r") as f:
            user_data = load(f)

        sorted_words = create_sorted_filtered_list(user_data)

        with open(user_ts_path, "w") as f:
            for word, timestamp, timings in sorted_words:
                f.write(f"{timestamp}\t{word}\t{len(word) - 1}\t{timings}\n")

        print(f"Finished writing to {user_ts_path}")


if __name__ == "__main__":
    main()

