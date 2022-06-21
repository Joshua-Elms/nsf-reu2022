from pathlib import PurePath
import os
from json import load

def word_conditions(content):
    word, _ = content
    is_word = True
    length = len(word)
    lower_incl = 2
    upper_incl = 15
    unique_letters = set(word)
    unique_total_min_ratio = .51

    # word inside length limits
    if length < lower_incl or length > upper_incl:
        is_word = False

    # ratio of unique:total letters must be higher than ratio to be considered real word
    elif (len(unique_letters) / length) < unique_total_min_ratio:
        is_word = False

    return is_word

def create_sorted_list(user_dict):
    word_occurences = [(key, user_dict[key]["timing_vectors"]) for key in user_dict]
    filtered_words = tuple(filter(word_conditions, word_occurences))
    sorted_filtered_words = sorted(filtered_words, key = lambda x: x[1], reverse = True)

    for i, word in enumerate(sorted_filtered_words):
        if i < 10:
            print(word)

    return sorted_filtered_words

def main():
    user_json_path = PurePath("../../data/user_json_files/")
    user_time_series_path = PurePath("../../data/user_time_series/")
    user_raw_path = PurePath("../../data/clarkson2_files/")
    user_list = [user for user in os.listdir(user_raw_path) if user != ".DS_Store"]

    user_paths = [PurePath(user_json_path, PurePath(f"user_{user}.json")) for user in user_list]

    all_user_sorted_words = []
    for i, user_path in enumerate(user_paths):
        with open(user_path, "r") as f:
            user_data = load(f)

        sorted_words = create_sorted_list(user_data)
        all_user_sorted_words.append(sorted_words)
        
    print(len(all_user_sorted_words))

if __name__ == "__main__":
    main()

