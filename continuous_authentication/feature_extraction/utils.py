def gen_graph_names(word):
    """
    A utility function to generate both monographs and digraph column names associated with a word
    Example: gen_graph_names("CAT") -> [""]
    """
    # count letters first
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

if __name__ == "__main__":
    print(gen_graph_names("CAT"))