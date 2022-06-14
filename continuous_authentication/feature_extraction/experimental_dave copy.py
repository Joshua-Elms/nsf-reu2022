from pathlib import PurePath
import os
import numpy as np
import csv

def not_first_or_last(i, sample):
    return not ( (i == 0) or (i == len(sample)-1) )

def previous_entry_is_letter(i, sample, letters):
    return sample[i-1] in letters

def parse_word(word_indices, raw_keystrokes):
    lines = raw_keystrokes[word_indices]
    

    return lines
word_indeces = []
def word_index(letters, sample_path):
    #down is 0, up is 1, so start of word has to to be = 0 with last 0 value is non letter 
    #end of word is true if last letter typed has value 1, get rid of each enter value 
    #if value is part in letters, add index to word_indeces
    #start parse if value before is non letter and current value is letter unless last value is 1
    word_indeces = []

    with open(sample_path, "r") as f: 
        rm_newline = lambda x: (x[0], x[1], x[2].rstrip("\n"))
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
        keystroke_arr = np.array(nested_keystrokes) 
    
    size = 9
    i = 0 
    for i in range(size):
        if (keystroke_arr[i,2] != letters):
            i+= 1
            
        if (keystroke_arr[i,2] in letters):
            word_indeces.append(i)
            i+= 1


    
    return word_indeces


    print()
def test_main():
    # c2_path = PurePath("../../data/clarkson2_files")
    # subject_84500_path_partial = PurePath("84500")
    # sample_path = PurePath(c2_path, subject_84500_path_partial)
    sample_path = PurePath("synthetic_data/synthetic_c2.txt")
    letters = [chr(i) for i in range(65, 91)]
    word_cnt = 0
    test_file = word_index(letters, sample_path)
    print(test_file)
    #print(os.getcwd())


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
    #print(parsed)
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
    

