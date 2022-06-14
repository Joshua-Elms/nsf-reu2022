from pathlib import PurePath
import os
import numpy as np

def not_first_or_last(i, sample):
    return not ( (i == 0) or (i == len(sample)-1) )

def previous_entry_is_letter(i, sample, letters):
    return sample[i-1] in letters

def parse_word(word_indices, raw_keystrokes):
    lines = raw_keystrokes[word_indices]
    

    return lines


def test_main():
    # c2_path = PurePath("../../data/clarkson2_files")
    # subject_84500_path_partial = PurePath("84500")
    # sample_path = PurePath(c2_path, subject_84500_path_partial)
    sample_path = PurePath("synthetic_data/synthetic_c2.txt")
    letters = [chr(i) for i in range(65, 91)]
    word_cnt = 0

    with open(sample_path, "r") as f: 
        rm_newline = lambda x: (x[0], x[1], x[2].rstrip("\n"))
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
        keystroke_arr = np.array(nested_keystrokes)
    
    # [print(nested_keystrokes[i]) for i in range(10)]
    # for i, event in enumerate(nested_keystrokes):
    #     if previous_entry_is_letter(i, nested_keystrokes, letters):
    #         if not_first_or_last(i, nested_keystrokes):
    #             # start recording word
    parsed = parse_word([1, 3, 4, 5, 6, 8], keystroke_arr)
    print(parsed)
    pass


def main():
    pass


if __name__ == "__main__":
    """
    List of features to be extracted:
        - Monographs
        - DU, DD, UD, UU Digraphs
        - Words
    """
    test_main()
    

