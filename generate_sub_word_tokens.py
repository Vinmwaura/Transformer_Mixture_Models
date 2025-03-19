import re
import glob
import json
import pathlib
import argparse
from collections import Counter

def get_most_frequent_pair(sub_words_list):
    # Define sub_words to exclude (numbers & punctuation).
    EXCLUDE_PATTERN = re.compile(r'[\d\W]')  # Matches digits (\d) and non-word chars (\W).

    # Find the most frequent adjacent pair, skipping excluded sub_words.
    sub_words_pairs = Counter()

    for index in range(len(sub_words_list) - 1):
        sub_word_token_1 = sub_words_list[index]
        sub_word_token_2 = sub_words_list[index + 1]

        # Skip pairs containing unwanted sub_words.
        if EXCLUDE_PATTERN.search(sub_word_token_1) or EXCLUDE_PATTERN.search(sub_word_token_2):
            continue

        sub_words_pairs[(sub_word_token_1, sub_word_token_2)] += 1

    return sub_words_pairs

def merge_pair(sub_words_list, frequent_sub_word):
    # Merge the most frequent pair into a single token.
    curr_index = 0
    next_index = curr_index + 1
    max_index = len(sub_words_list) - 1

    while curr_index < max_index:
        curr_sub_word = sub_words_list[curr_index]
        next_sub_word = sub_words_list[next_index]

        if (curr_sub_word,next_sub_word) == frequent_sub_word:
            temp_next_sub_word = sub_words_list.pop(next_index)
            sub_words_list[curr_index] = sub_words_list[curr_index] + temp_next_sub_word
            max_index = len(sub_words_list) - 1
            continue

        curr_index += 1
        next_index += 1

    return sub_words_list

# Type function for argparse - a float within some predefined bounds.
def range_limited_float_type(arg):
    MIN_VAL = 0.1
    MAX_VAL = 0.9
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
    return f

def main():
    parser = argparse.ArgumentParser(
        description="Generates Sub-Word JSON file.")

    parser.add_argument(
        "--dataset-path",
        help="Filepath to dataset.",
        required=True,
        type=str)
    parser.add_argument(
        "--vocab-size",
        help="Max size of vocabulary.",
        type=int,
        default=2_048)
    parser.add_argument(
        "--out-path",
        help="Destination output path for json output.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--dataset-ratio",
        help="Ratio of train dataset, for splitting dataset.",
        type=range_limited_float_type,
        default=0.8)

    args = vars(parser.parse_args())

    vocab_size = args["vocab_size"]
    dataset_path = args["dataset_path"]  # Regex to dataset.
    out_path = args["out_path"]  # Output Json file e.g ./<folder_path>/train.json.
    dataset_ratio = args["dataset_ratio"]  # Ratio of train:test dataset.

    with open(dataset_path, "r", encoding='utf-8-sig') as file:
        text = file.read()

    sub_words_list = list(text)
    print(f"Length of sub-words: {len(sub_words_list):,}")

    vocabs = list(set(sub_words_list))
    len_vocabs = len(vocabs)

    sorted_vocabs = sorted(vocabs)
    print(f"Vocab:\n{sorted_vocabs}")
    print(f"Length of unique vocab:\n{len_vocabs:,}")

    loop = True
    while loop:
        if len_vocabs >= vocab_size:
            loop = False
            break

        print("*" * 100)

        sub_word_counter = get_most_frequent_pair(sub_words_list=sub_words_list)
        most_common_data = sub_word_counter.most_common(1)

        if len(most_common_data) > 0:
            frequent_sub_word, count = most_common_data[0]
            print(f"Most frequent pair tokens: {frequent_sub_word}, Count: {count:,}")

            # TODO: Use multiprocessing and better merging algorithm for speed.
            sub_words_list = merge_pair(
                sub_words_list=sub_words_list,
                frequent_sub_word=frequent_sub_word)

            vocabs = list(set(sub_words_list))
            sorted_vocabs = sorted(vocabs)
            len_vocabs = len(sorted_vocabs)

            print(f"Length of unique vocab: {len_vocabs:,}")
            print(f"Length of sub-words: {len(sub_words_list):,}")
        else:
            loop = False
            break

        print("*" * 100)

    sorted_vocabs = sorted(vocabs)
    print(f"Vocab:\n{sorted_vocabs}")

    sub_words_indices_list = []
    for sub_word in sub_words_list:
        sub_words_indices_list.append(sorted_vocabs.index(sub_word))

    train_dataset_length = round(len(sub_words_indices_list) * dataset_ratio)

    data_json = {
        "vocab": sorted_vocabs,
        "all": sub_words_indices_list,
        "train": sub_words_indices_list[:train_dataset_length],
        "test": sub_words_indices_list[train_dataset_length:]}

    with open(out_path, "w") as json_f:
        json.dump(data_json, json_f)

if __name__ == "__main__":
    main()
