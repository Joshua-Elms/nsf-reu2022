from pathlib import PurePath
import os

from zmq import PUB 

def main(**kwargs):
    pass


if __name__ == "__main__":
    """
    List of features to be extracted:
        - Monographs
        - DU, DD, UD, UU Digraphs
        - Words
        - wpm (maybe)
        - Other n-graphs (maybe)
    """
    main()

    c2_path = PurePath("../../data/clarkson2_files")
    subject_84500_path_partial = PurePath("84500")
    subject_84500_path = PurePath(c2_path, subject_84500_path_partial)
    with open(subject_84500_path, "r") as f: 
        for line in f.readlines():
            print(line)

