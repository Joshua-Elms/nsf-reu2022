from pathlib import PurePath
import os
import numpy as np

def not_first_or_last(i, sample):
    return not ( (i == 0) or (i == len(sample)-1) )

def previous_entry_is_letter(i, sample, letters):
    return sample[i-1] in letters

def parse_word(word_indices, raw_keystrokes):
    lines = tuple(raw_keystrokes[i] for i in word_indices)

    letter_counter = [0 for _ in range(26)]
    order_dict = {}
    for i, line in enumerate(lines):
        timestamp_str, release_str, letter = [line[i] for i in range(3)]
        timestamp, release = int(timestamp_str), int(release_str)
        letter_idx = ord(letter) - 65

        if not release: # key was pressed
            letter_counter[letter_idx] += 1
            order_dict[f"{letter}{letter_counter[letter_idx]}"] = [lines[i]] # [i] for index instead of value

        else: # key not pressed
            order_dict[f"{letter}{letter_counter[letter_idx]}"] += (lines[i]) # [i] same

        
    return order_dict

def create_json():
    x = {
        'time_stamp': None,
        'up_or_down': None,
        'Letter': None
    }
    data = []


    print(x)
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
    
    # [print(nested_keystrokes[i]) for i in range(10)]

    # testing "cat"
    parsed = parse_word([1, 3, 4, 5, 6, 8], nested_keystrokes)

    # # testing "addressable"
    # parsed = parse_word([i for i in range(11, 32)] + [33], nested_keystrokes)
    print(parsed)
    # [print(line) for line in parsed]
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
    

