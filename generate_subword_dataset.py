import json
import pathlib
import argparse

import torch

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

def generate_dataset(device, vocab, subwords_list):
    # Stride for sliding window.
    stride = 1

    # Dictionary of vocabulary indices for quick lookup.
    vocab_index = {}
    for index, vocab_item in enumerate(vocab):
        vocab_index[vocab_item] = index

    # Get indices of characters from Vocabulary.
    subwords_indices = [vocab_index[subword] for subword in subwords_list]

    # Sort Vocabulary by length in descending order.
    sorted_vocabs = sorted(
        vocab,
        key=len,
        reverse=True)

    for index, sorted_vocab in enumerate(sorted_vocabs):
        print("=" * 100)
        print(f"{index + 1:,} / {len(sorted_vocabs):,} | Processing: {repr(sorted_vocab)} | Dataset size: {len(subwords_indices):,}")

        # Break down Vocabulary item into individual characters.
        vocab_character_list = list(sorted_vocab)
        vocab_character_indices = [vocab_index[vocab_character] for vocab_character in vocab_character_list]

        # Length of Vocabulary item being searched for.
        window_size = len(vocab_character_indices)

        # Skip single characters.
        if window_size < 2:
            continue

        vocab_character_indices_tensor = torch.tensor(
            vocab_character_indices,
            device=device)  # (num_vocab_characters,)

        subwords_indices_tensor = torch.tensor(
            subwords_indices,
            device=device)  # (num_subwords,)

        # Simulate sliding window operation all at once.
        subwords_indices_tensor_unfold = subwords_indices_tensor.unfold(
            dimension=0,
            size=window_size,
            step=stride)  # (num_subwords_pairs,window_size)

        # Boolean mask where new subword pair matches in subword.
        match_mask_tensor = torch.eq(
            subwords_indices_tensor_unfold,
            vocab_character_indices_tensor).all(dim=1)  # (num_subword_pairs,)

        # Convert mask and subword_indices to list.
        match_mask_list = match_mask_tensor.tolist()
        subwords_indices_list = subwords_indices_tensor.tolist()

        new_subword_index = vocab.index(sorted_vocab)

        subwords_indices = merge_pair(
            window_size=window_size,
            new_subword_index=new_subword_index,
            match_mask_list=match_mask_list,
            subwords_indices_list=subwords_indices_list)

    return subwords_indices

def main():
    parser = argparse.ArgumentParser(
        description="Generates subword token dataset.")

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
        type=pathlib.Path)
    parser.add_argument(
        "--vocab-path",
        help="Filepath to vocabulary.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-path",
        help="Destination output path for json file.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]
    dataset_path = args["dataset_path"]
    vocab_path = args["vocab_path"]
    out_path = args["out_path"]

    # Load Vocabulary json and text file.
    with open(vocab_path, "r") as vocab_f, open(dataset_path, "r", encoding='utf-8-sig') as txt_f:
        text = txt_f.read()
        vocab_json = json.load(vocab_f)

    vocab = vocab_json["vocabulary"]
    subwords_list = list(text)

    subwords_indices_list = generate_dataset(
        device=device,
        vocab=vocab,
        subwords_list=subwords_list)

    # Sanity check to ensure at no point deviations occured.
    new_text = "".join([vocab[subword_index] for subword_index in subwords_indices_list])
    is_valid = (new_text == text)
    print(f"Is data valid? {is_valid}")

    data_dict = {
        "vocabulary": vocab,
        "data": subwords_indices_list
    }

    with open(out_path, "w") as f:
        json.dump(data_dict, f)

if __name__ == "__main__":
    main()
