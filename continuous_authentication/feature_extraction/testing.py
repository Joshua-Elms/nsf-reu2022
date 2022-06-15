import numpy as np
from pathlib import PurePath

if __name__ == "__main__":
    c2_path = PurePath("../../data/clarkson2_files")
    subject_84500_path_partial = PurePath("97562")
    sample_path = PurePath(c2_path, subject_84500_path_partial)
    # sample_path = PurePath("synthetic_data/synthetic_c2.txt")

    with open(sample_path, "r") as f: 
        rm_newline = lambda x: (int(x[0]), int(x[1]), x[2].rstrip("\n"))
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
        keystroke_arr = np.array(nested_keystrokes)

    print(np.unique(keystroke_arr[:, -1]))