from ast import expr_context
from pathlib import PurePath
import os
import numpy as np
import sys
sys.path.append("/Users/joshuaelms/Desktop/github_repos/nsf-reu2022/continuous_authentication")

from experimental import file_to_nested_tuples

def determine_is_gamer(keystrokes):
    letters = [val[2] for i, val in enumerate(keystrokes)]

    gamer_keys = list("WASD")
    letters = [chr(i) for i in range(65, 91)]
    gamer_key_cnt = 0
    longest_streak = 0
    longest_pos = 0
    current_streak = 0

    for i in range(1, len(letters)):
        letter = letters[i]
        if letter in gamer_keys and letters[i - 1] in gamer_keys:
            current_streak += 1
        
        else:
            if current_streak > longest_streak:
                longest_streak = current_streak
                longest_pos = i

    return (longest_pos, longest_streak)

def find_gamer():
    c2 = PurePath("../../data/clarkson2_files")
    users = os.listdir()
    gamer_scores = []
    for user in users:
        try:
            int(user)
        
        except ValueError:
            continue

        path = PurePath(c2, PurePath(user))
        keystrokes = file_to_nested_tuples(path)
        # print(keystrokes[0])    
        is_gamer = determine_is_gamer(keystrokes)
        gamer_scores.append((user, *is_gamer))

    gamer_scores = np.array(gamer_scores).astype("float64")
    maximum_gamer_index = np.argmax(gamer_scores[:, 2])
    result = gamer_scores[maximum_gamer_index]
    print(result)


if __name__ == "__main__":
    find_gamer()