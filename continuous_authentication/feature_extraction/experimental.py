from pathlib import PurePath
import os

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

def file_to_nested_tuples(path):
    with open(path, "r") as f: 
        rm_newline = lambda x: (int(x[0]), int(x[1]), x[2].rstrip("\n"))
        nested_keystrokes = tuple(rm_newline(line.split("\t")) for line in f.readlines())

    return nested_keystrokes

def clean_text(keystrokes):
    letters = [chr(i) for i in range(65, 91)]
    result = [(arr[1], arr[2]) for arr in keystrokes]
    ignored = ["LShiftKey", "RShiftKey"]
    added_sep = False
    out_lst = []

    for i, char in enumerate(result):
        is_release, letter = char
        if letter in letters:
            out_lst += [i]

            if added_sep:
                added_sep = False
        
        elif letter in ignored:
            continue

        else:
            if not added_sep and not is_release:
                out_lst += [-1]
                added_sep = True

    return out_lst

def get_word_positions(keystrokes):
    pressed = {}
    in_word = False
    all_words = []
    num_keystrokes = len(keystrokes)
    start = 0
    current_word = []
    sep = "-"
    last_index_reached = 0

    while last_index_reached < num_keystrokes-1:
        for i in range(start, num_keystrokes):
            last_index_reached = i
            time, is_release, letter = keystrokes[i]

            if not in_word and not pressed:
                if current_word:
                    # done pressing all keys, haven't yet added to all_words
                    all_words.append(current_word)
                    current_word = []
                    break

                else:
                    if letter != sep and not is_release:
                        # begin condition
                        in_word = True
                        break

                    else:
                        # skip separator
                        start += 1
                        break

            elif not in_word and pressed:
                # waiting for last letter release
                if letter in pressed and is_release:
                    current_word.append(pressed.pop(letter))
                    current_word.append(i)

                else: continue

            else:
                if letter != sep:
                    # detected non-separator
                    if not is_release:
                        # add pressed key to pressed
                        pressed[letter] = i

                    else: 
                        # search for released key in pressed; if found, add to current_word
                        if letter in pressed:
                            current_word.append(pressed.pop(letter))
                            current_word.append(i)

                        else: continue
                else: 
                    # detected separator
                    in_word = False
                    start = i # + 1

    if current_word:
        all_words.append(current_word)

    return all_words

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

def process_sample(nested_keystrokes):
    """
    Read in tabular 3-field keystroke data then extract the words and their n-graphs from the data

    Args: 
        sample_path <PurePath>: where to read the raw data from
        slice <tuple(lower (int, upper (int))>: range of lines to process

        
    Returns: 
        sample_contents <dict>: same format as persistent JSON profiles
    """
    cleaned_indices = clean_text(nested_keystrokes)
    
    cleaned_keystrokes = [nested_keystrokes[i] if i >= 0  else (None, None, "-") for i in cleaned_indices]
    
    indices_of_all_words = get_word_positions(cleaned_keystrokes)

    sample_contents = {}
    for single_word_indices in indices_of_all_words:
        word, time, graphs = parse_word(single_word_indices, cleaned_keystrokes)

        if word in sample_contents:
            sample_contents[word].append([time, graphs])
        
        else:
            sample_contents[word] = [[time, graphs]]

    return sample_contents

def test_many():
    c2_path = PurePath("../../data/clarkson2_files")
    sample_path = PurePath("synthetic_data/synthetic_c2.txt")
    files = os.listdir(c2_path)

    for i, file in enumerate(files):
        try:
            int(file)

        except ValueError:
            continue

        if int(file) != 991447:
            break
        partial = PurePath(file)
        full_path = PurePath(c2_path, partial)
        keystrokes = file_to_nested_tuples(full_path)
        output = process_sample(keystrokes)
        print("-" * 50)
        print(f"File Name: {file}\nFile Rank: {i}/{len(files)}")
        for i, item in enumerate(output.items()):
            key, val = item
            if i < 200:
                print(key) 

    pass
    
if __name__ == "__main__":
   test_many()

