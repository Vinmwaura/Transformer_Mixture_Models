import os
import json
import pathlib
import logging
import argparse

import torch

def get_most_frequent_pair(subwords_indices_unfold):
    # Returns unique subwords and their counts.
    unique_subwords, unique_subwords_counts = torch.unique(
        subwords_indices_unfold,
        dim=0,
        return_counts=True)

    max_count_index = torch.argmax(unique_subwords_counts)

    max_count = unique_subwords_counts[max_count_index].item()
    most_frequent_pair = unique_subwords[max_count_index]

    return max_count, most_frequent_pair

def merge_pair(
        window_size,
        new_subword_index,
        match_mask_list,
        subwords_indices_list):
    new_subwords_indices = []

    # Number used to determine how many subword need to be skipped during iteration.
    skip_count = 0

    # Iterate over subword indices and combine the most frequent subword pairs into a new subword.
    for subword_index, is_match in enumerate(match_mask_list):
        not_skip_flag = skip_count <= 0

        if not_skip_flag and is_match:
            new_subwords_indices.append(new_subword_index)
            skip_count = (window_size - 1)
        elif not_skip_flag and not is_match:
            new_subwords_indices.append(subwords_indices_list[subword_index])
        else:
            # Implemented this way to avoid list manipulation as it's slow.
            skip_count = skip_count - 1

    if not is_match:
        # Insert the final subword indices not represented in the loop.
        new_subwords_indices.extend(subwords_indices_list[subword_index + 1:])

    return new_subwords_indices

def main():
    parser = argparse.ArgumentParser(
        description="Generate subword Vocabulary from text file.")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--dataset-path",
        help="Filepath to text dataset.",
        required=True,
        type=str)
    parser.add_argument(
        "--vocab-size",
        help="Max size of vocabulary.",
        type=int,
        default=2_048)
    parser.add_argument(
        "--out-path",
        help="Destination output path for json dictionary.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]
    vocab_size = args["vocab_size"]
    dataset_path = args["dataset_path"]
    out_path = args["out_path"]

    try:
        os.makedirs(out_path, exist_ok=True)
    except Exception as e:
        raise e
    
    # Log file path.
    log_path = os.path.join(
        out_path,
        "vocabulary.log")

    # Logs Info to parent directory.
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG,
        force=True)

    # Dictionary parameters.
    stride = 1
    window_size = 2

    with open(dataset_path, "r", encoding='utf-8-sig') as file:
        text = file.read()

    # Separate text into a list of individual characters(Alphanumerical + Special character)
    subwords = list(text)

    # Get unique characters
    vocabs = list(set(subwords))
    sorted_vocabs = sorted(vocabs)

    # Initial size of Vocabulary.
    len_vocabs = len(sorted_vocabs)

    # Dictionary of vocabulary indices for quick lookup.
    vocab_index = {}
    for index, vocab_item in enumerate(sorted_vocabs):
        vocab_index[vocab_item] = index

    # Get indices of subwords from sorted vocabulary.
    subwords_indices_list = [vocab_index[subword] for subword in subwords]

    count_loop = 0
    max_count_loop = 100_000

    if vocab_size > max_count_loop:
        raise Exception("Vocab size is too large!")

    logging.info("Generating subword Vocabulary.")
    while True:
        if len_vocabs >= vocab_size:
            break

        logging.info("*" * 100)
        logging.info(f"Length of unique vocab: {len_vocabs:,}")
        logging.info(f"Length of sub-words: {len(subwords_indices_list):,}")

        # Convert list of integers into tensors.
        subwords_indices_tensor = torch.tensor(
            subwords_indices_list,
            dtype=torch.long,
            device=device)  # (N,)

        # Simulate sliding window operation all at once.
        subwords_indices_unfold = subwords_indices_tensor.unfold(
            dimension=0,
            size=window_size,
            step=stride)  # (num_subwords_pairs,window_size)

        # Get most frequent subword pairs and it's total count.
        max_count, most_frequent_pair = get_most_frequent_pair(
            subwords_indices_unfold=subwords_indices_unfold)

        # Combine most frequent subword pairing to form a new subword.
        frequent_pair_list = most_frequent_pair.tolist()
        subword_string_list = [sorted_vocabs[i] for i in frequent_pair_list]
        new_subword_string = "".join(subword_string_list)

        logging.info(f"Most frequent subwords pairings: {subword_string_list} | Count: {max_count:,}")

        # Append new subword to Vocabulary.
        sorted_vocabs.append(new_subword_string)

        # Get index of new subword at end of Vocabulary.
        new_subword_index = len(sorted_vocabs) - 1

        # Boolean mask where new subword pair matches in subword.
        match_mask_tensor = torch.eq(
            subwords_indices_unfold,
            most_frequent_pair).all(dim=1)  # (num_subword_pairs,)

        # Convert mask and subword_indices to list.
        match_mask_list = match_mask_tensor.tolist()
        subwords_indices_list = subwords_indices_tensor.tolist()

        # Merge most frequent subword pairs to form one subword.
        subwords_indices_list = merge_pair(
            window_size=window_size,
            new_subword_index=new_subword_index,
            match_mask_list=match_mask_list,
            subwords_indices_list=subwords_indices_list)

        # Get list of unique indices from new subword list.
        vocabs_indices = list(set(subwords_indices_list))
        sorted_vocabs_indices = sorted(vocabs_indices)

        # Recompute to remove subwords not being utilized after merge operation.
        sorted_vocabs = [sorted_vocabs[sorted_vocabs_index] for sorted_vocabs_index in sorted_vocabs_indices]
        len_vocabs = len(sorted_vocabs)  # New length of Vocabulary.

        count_loop += 1

        # HACK: Prevent infinite loops in cases of wierd issues.
        if count_loop >= max_count_loop:
            logging.info("Reached maximum iteration allowed!")
            break

    logging.info("*" * 100)
    logging.info(f"Length of unique vocab: {len_vocabs:,}")
    logging.info(f"Length of sub-words: {len(subwords_indices_list):,}")
    logging.info("*" * 100)

    # Sanity check to ensure at no point deviations occured.
    new_text = "".join([sorted_vocabs[subword_index] for subword_index in subwords_indices_list])
    is_valid = (new_text == text)
    logging.info(f"Is data valid? {is_valid}")

    vocabulary = {"vocabulary": sorted_vocabs}
    data_dict = {"vocabulary": sorted_vocabs, "data": subwords_indices_list}

    vocabulary_path = os.path.join(
        out_path,
        "vocabulary.json")
    data_path = os.path.join(
        out_path,
        "data.json")
    with open(vocabulary_path, "w") as f, open(data_path, "w") as g:
        json.dump(vocabulary, f)
        json.dump(data_dict, g)

if __name__ == "__main__":
    main()
