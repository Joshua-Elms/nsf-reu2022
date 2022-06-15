from pathlib import PurePath
import os


def not_first_or_last(i, sample):
    return not ( (i == 0) or (i == len(sample)-1) )

def previous_entry_is_letter(i, sample, letters):
    return sample[i-1] in letters

def parse_word(word_indices, raw_keystrokes):
    """
    Args: 
        word_indices: list of indices that comprise every key press/release in a given word in the raw keystroke data
        raw_keystrokes: [[keystroke_event_1],...,[keystroke_event_n]]; those events contain timestamp (int), press/release (int/bool), and key pressed (str)

    Returns: 
        An iterable object containing the word typed (str), the timestamp of the first keypress in the word (int), and the timing vector for the word (list)
    """
    lines = tuple(raw_keystrokes[i] for i in word_indices) # pulling out keystrokes that make up word

    ### Determine the order each of each keypress in word ###
    letter_counter = [0 for _ in range(26)] 
    order_dict = {} # order of letters typed is stored in case user releases keys in different order than they press them
    for i, line in enumerate(lines):
        timestamp, release, letter = [line[i] for i in range(3)]
        letter_idx = ord(letter) - 65

        if not release: # key was pressed
            letter_counter[letter_idx] += 1
            order_dict[f"{letter}{letter_counter[letter_idx]}"] = [lines[i]] # [i] for index instead of value

        else: # key not pressed
            order_dict[f"{letter}{letter_counter[letter_idx]}"] += [lines[i]] # [i] same

    ### Calculate monographs and digraphs associated with word ###
    dict_keys = [key for key in order_dict.keys()]
    num_letters = len(dict_keys)
    digraphs = [] 
    monographs = []
    for i, key in enumerate(dict_keys):
        this_letter = order_dict[key]
        this_d, this_u = this_letter
        mono = this_u[0] - this_d[0]
        monographs.append(mono)

        if i < num_letters - 1:
            next_letter = order_dict[dict_keys[i + 1]]
            next_d, next_u = next_letter
            digraphs.append(next_d[0] - this_d[0]) # dd
            digraphs.append(next_u[0] - this_d[0]) # du
            digraphs.append(next_d[0] - this_u[0]) # ud
            digraphs.append(next_u[0] - this_u[0]) # uu
        
    ### Combine results and prep for output ###
    word = "".join([key[0] for key in dict_keys])
    all_graphs = monographs + digraphs
    timestamp_first_keypress = lines[0][0]
    word_info = (word, timestamp_first_keypress, all_graphs)

    return word_info


def gen_graph_names(word):
    """
    A utility function to generate both monographs and digraph column names associated with a word
    Example: gen_graph_names("CAT") -> ['m_C1', 'm_A1', 'm_T1', 'DD_C1A1', 'DU_C1A1', 'UD_C1A1', 'UU_C1A1', 'DD_A1T1', 'DU_A1T1', 'UD_A1T1', 'UU_A1T1']

    Args: word <str>
    Returns: graph_names <list[str, ... str]>
    """
    # letters occurences are counted to make them distinct; "BANANA" has three "A"s, and they might all be typed differently
    letter_counter = [0 for _ in range(26)]
    subscripted_word = []
    for letter in word:
        letter_idx = ord(letter) - 65
        letter_counter[letter_idx] += 1
        subscripted_word += [f"{letter}{letter_counter[letter_idx]}"]

    monograph_labels = [f"m_{letter}" for letter in subscripted_word]
    digraph_labels_incomplete = [f"{subscripted_word[i]}{subscripted_word[i+1]}" for i in range(len(word) - 1)]
    digraph_labels = [f"{pre}_{post}" for post in digraph_labels_incomplete for pre in ["DD", "DU", "UD", "UU"]]
    col_names = monograph_labels + digraph_labels

    return col_names


def process_sample():
    """
    Read in tabular 3-field keystroke data then extract the words and their n-graphs from the data

    Args: 
        sample_path <PurePath>: where to read the raw data from
        slice <tuple(lower (int, upper (int))>: range of lines to process
        
        
    Returns: 
        sample_contents <dict>: same format as persistent JSON profiles
    """
    # c2_path = PurePath("../../data/clarkson2_files")
    # subject_84500_path_partial = PurePath("84500")
    # sample_path = PurePath(c2_path, subject_84500_path_partial)
    sample_path = PurePath("synthetic_data/synthetic_c2.txt")
    letters = [chr(i) for i in range(65, 91)]
    word_cnt = 0

    with open(sample_path, "r") as f: 
        rm_newline = lambda x: (int(x[0]), int(x[1]), x[2].rstrip("\n"))
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())
    
    # [print(nested_keystrokes[i]) for i in range(10)]

    # indices_of_all_words = find_words()
    indices_of_all_words = [[1, 3, 4, 5, 6, 8]]#, [i for i in range(11, 32)] + [33]]

    output_dict = {}
    for single_word_indices in indices_of_all_words:
        word, time, graphs = parse_word(single_word_indices, nested_keystrokes)
        output_dict[word] = [time, graphs]

    print(output_dict)
    for key, val in output_dict.items():
        print(len(val[1]))
        

    # # testing "cat"
    # parsed = parse_word([1, 3, 4, 5, 6, 8], nested_keystrokes)

    # # testing "addressable"
    # parsed = parse_word([i for i in range(11, 32)] + [33], nested_keystrokes)
    # print(parsed)
    # [print(line) for line in parsed]
    pass

if __name__ == "__main__":
    """
    List of features to be extracted:
        - Monographs
        - DU, DD, UD, UU Digraphs
        - Words
    """
    process_sample()
    

