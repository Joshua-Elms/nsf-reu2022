from cmath import e
from os import remove
import numpy as np
from pathlib import PurePath

def remove_crap(nested_keystrokes):
    letters = [chr(i) for i in range(65, 91)]
    result = [arr[2] for arr in nested_keystrokes]
    ignored = ["LShiftKey", "RShiftKey"]
    added_sep = False
    out_lst = []
    for i, char in enumerate(result):
        if char in letters:
            out_lst += char

            if added_sep:
                added_sep = False
        
        elif char in ignored:
            continue

        else:
            if not added_sep:
                out_lst += "-"
                added_sep = True

    cnt = 0
    with open("just_letters.txt", "w") as f:
        for i, letter in enumerate(out_lst):
            f.write(letter)
            cnt += 1
            if cnt == 30:
                cnt = 0
                f.write('\n')


    # result = [(arr[0], arr[1]) if arr[1] in letters else "" for arr in result]
    # result = [i[0] if not type(i) == tuple else i for i in result]
    # result = [nested_keystrokes[i][-1] for i in result]

    # print(result)

    pass

if __name__ == "__main__":
    c2_path = PurePath("../../data/clarkson2_files")
    subject_84500_path_partial = PurePath("97562")
    sample_path = PurePath(c2_path, subject_84500_path_partial)
    # sample_path = PurePath("synthetic_data/synthetic_c2.txt")

    with open(sample_path, "r") as f: 
        rm_newline = lambda x: (int(x[0]), int(x[1]), x[2].rstrip("\n"))
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
        keystroke_arr = np.array(nested_keystrokes)

    letters = [chr(i) for i in range(65, 91)]
    
    remove_crap(nested_keystrokes)
    # cnt = 0
    # with open("just_letters.txt", "w") as f:
    #     for i, letter in enumerate(result):
    #         f.write(letter)
    #         cnt += 1
    #         if cnt == 30:
    #             cnt = 0
    #             f.write('\n')
    # print(result)

    # print(np.unique(keystroke_arr[:, -1]))