from pathlib import PurePath
import os

def test_main():
    c2_path = PurePath("../../data/clarkson2_files")
    subject_84500_path_partial = PurePath("84500")
    subject_84500_path = PurePath(c2_path, subject_84500_path_partial)

    with open(subject_84500_path, "r") as f: 
        rm_newline = lambda x: (x[0], x[1], x[2][:-1])
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
        print(nested_keystrokes[:5])

            

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
    

