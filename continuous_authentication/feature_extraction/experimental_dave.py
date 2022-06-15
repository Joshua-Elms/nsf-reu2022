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
    non_letters = []
    print(sample_path)
    shape = keystroke_arr.shape
    print("shape: ", shape)
    size = shape[0]
    print("size: ", size)
    i = 1 
    non_letter = 0
    total_dict= []
    #loops until you record 2 key releases
    end_word = non_letter / 2



    for i in range(size):
        #need to record all words not just first


        #we need to seperate after each even count for this algorith
        #Once it reaches an even count we want to take that list of indexes and add them to an overall list.
        #Each list will have the index's of the keystrokes for the word. seperate list for each different word. 
        if(non_letter % 2) == 0:
            #while amount of key releases of non letters is not equal to 2, keep recording word
            #if(end_word % 2) == 0:
            #if(non_letter != 2):
                #records non letter key releases
            if (keystroke_arr[i,2] not in letters):
                if (keystroke_arr[i,1] == '1'):
                        non_letters.append(i)
                        i+= 1
                        non_letter += 1
                #until non_letters size is equal to 2 you will add all the indexes to word profile
            else :
                word_indeces.append(i)
                
            

    

        
    #print("word count: ", test_count)
    print("total dict test: ", total_dict)
    print("non letter test:", non_letters)

    
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
    print("index values: ",test_file)
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

    
    parsed = parse_word(test_file, keystroke_arr)
    print("Parsed: ",parsed)
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
    

