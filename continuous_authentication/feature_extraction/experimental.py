from pathlib import PurePath
import os

def test_main():
    # c2_path = PurePath("../../data/clarkson2_files")
    # subject_84500_path_partial = PurePath("84500")
    sample_path = PurePath("../../data/synthetic_c2.txt")

    with open(sample_path, "r") as f: 
        rm_newline = lambda x: (x[0], x[1], x[2][:-1])
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
    
    for i, event in enumerate(nested_keystrokes):


            

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
    

