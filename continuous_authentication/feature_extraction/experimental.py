from pathlib import PurePath
import os

def not_first_or_last(i, sample):
    return not ( (i == 0) or (i == len(sample)-1) )

def previous_entry_is_letter(i, sample, letters):
    return sample[i-1] in letters



def test_main():
    # c2_path = PurePath("../../data/clarkson2_files")
    # subject_84500_path_partial = PurePath("84500")
    sample_path = PurePath("../../data/synthetic_c2.txt")
    letters = [ord]

    with open(sample_path, "r") as f: 
        rm_newline = lambda x: (x[0], x[1], x[2][:-1])
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
    
    for i, event in enumerate(nested_keystrokes):
        if previous_entry_is_letter(i, nested_keystrokes, letters)
            if not_first_or_last(i, nested_keystrokes):

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
    

