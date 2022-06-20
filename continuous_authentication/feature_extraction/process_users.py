from pathlib import PurePath
import os
import json
from time import perf_counter as pfc

import sys
sys.path.append("../../")

from parse_utils import *


def write_batch_to_json():
    c2_path = PurePath("../../data/clarkson2_files")
    user_profile_folder = PurePath("../../data/processed_c2_data")
    files = os.listdir(c2_path)

    times = []
    for i, file in enumerate(files):
        start = pfc()
        user_path = PurePath(user_profile_folder, PurePath(f"user_{file}.json"))
        if file == ".DS_Store":
            continue

        partial = PurePath(file)
        full_path = PurePath(c2_path, partial)
        keystrokes = file_to_nested_tuples(full_path)
        output = process_sample(keystrokes)
        with open(user_path, "w") as f:
            json.dump(output, f, indent=4)

        stop = pfc()
        times.append(stop - start)

        print(f"Created JSON for user {file}")

    avg_parse_time = sum(times)/len(times)

    return avg_parse_time

if __name__ == "__main__":
    time = write_batch_to_json()
    print(f"Average parse time for all users is: {time}")