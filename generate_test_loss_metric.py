import os
import csv
import glob
import json
import math
import random
import pathlib
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Mixture_of_Models import MixtureofModels
from models.Mixture_of_Blocks import MixtureofBlocks

from dataset_loader.subword_dataset import SubWord_Dataset

from utils.model_utils import load_model

def main():
    project_name = "Test Loss"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--vocab-dataset-path",
        help="File path to Vocab json dataset file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--tst-dataset-path",
        help="File path to testing csv dataset file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--batch-size",
        help="Batch size of dataset.",
        type=int,
        default=64)
    parser.add_argument(
        "--model-checkpoints",
        help="Folder path to model checkpoints.",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to JSON config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    tst_dataset_path = args["tst_dataset_path"]  # Training csv file path (*.csv).
    vocab_dataset_path = args["vocab_dataset_path"]  # Vocabulary json file path (*.json).
    batch_size = args["batch_size"]  # Batch size of training dataset.
    model_checkpoints = args["model_checkpoints"]  # Folder paths to models saved.
    out_dir = args["out_dir"]  # Destination path for model.

    out_csv_file = os.path.join(
        out_dir,
        "loss.csv")

    config_json = args["config_path"]  # Load and Parse config JSON.
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    # Model Params (From config file).
    model_lr = config_dict["model_lr"]
    num_heads = config_dict["num_heads"]
    num_blocks = config_dict["num_blocks"]
    hidden_dim = config_dict["hidden_dim"]
    num_mixture = config_dict["num_mixture"]
    mixture_type = config_dict["mixture_type"]
    embedding_dim = config_dict["embedding_dim"]
    context_window = config_dict["context_window"]
    activation_type = config_dict["activation_type"]

    # Load JSON dataset.
    with open(vocab_dataset_path, "r") as json_f:
        vocab_json_dataset = json.load(json_f)

    # Vocabulary / Vocabulary size of NLP dataset.
    vocab = vocab_json_dataset["vocab"]
    vocab_size = len(vocab)

    # TODO: Save this in vocab dictionary.
    start_token = vocab_size
    padding_token = vocab_size + 1

    # Datasets.
    tst_dataset = SubWord_Dataset(
        csv_fpath=tst_dataset_path,
        start_token=start_token,
        padding_token=padding_token,
        context_window=context_window)

    # Dataloaders.
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True)

    if mixture_type == "models":
        mixture_model = MixtureofModels(
            num_models=num_mixture,
            num_embeddings=vocab_size + 2,  # Includes [START] and [PAD] tokens.
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            out_classes=vocab_size + 1,  # Includes only [PAD] tokens.
            use_checkpoint=False,
            activation_type=activation_type)
    elif mixture_type == "blocks":
        mixture_model = MixtureofBlocks(
            num_mixture=num_mixture,
            num_embeddings=vocab_size + 2,  # Includes [START] and [PAD] tokens.
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            out_classes=vocab_size + 1,  # Includes only [PAD] tokens.
            use_checkpoint=False,
            activation_type=activation_type)
    else:
        raise Exception("Invalid Mixture type!")

    model_regex = os.path.join(
        model_checkpoints,
        "*.pt")
    model_paths = glob.glob(model_regex)

    for index, model_path in enumerate(model_paths):
        print(f"Model: {index + 1:,} / {len(model_paths):,}, model_path: {model_path}")

        file_name = os.path.basename(model_path)

        model_index = file_name.split("_")[0]
        model_index = int(model_index)

        classifier_status, classifier_dict = load_model(model_path)
        if not classifier_status:
            raise Exception("An error occured while loading model checkpoint!")

        mixture_model.custom_load_state_dict(classifier_dict["model"])
        mixture_model = mixture_model.to(device)

        test_loss = []
        for index, (in_seq, target_seq) in enumerate(tst_dataloader):
            # Testing Data.
            in_seq = in_seq.to(device)  # (N,Seq)
            target_seq = target_seq.to(device)  # (N,Seq)

            mixture_model.eval()

            with torch.no_grad():
                out_classifier = mixture_model(in_seq)

                target_seq_flat = target_seq.flatten()  # (N*Seq,)
                out_classifier_flat = out_classifier.flatten(
                    start_dim=0,
                    end_dim=1)  # (N*Seq,Class)

                tst_classifier_loss = F.cross_entropy(
                    out_classifier_flat,
                    target_seq_flat,
                    ignore_index=len(vocab))
                tst_classifier_loss = tst_classifier_loss.item()

                test_loss.append(tst_classifier_loss)

        test_avg_loss = sum(test_loss) / len(test_loss)

        with open(out_csv_file, "a", newline="") as f:
            file_writer = csv.writer(f, delimiter=',')
            file_writer.writerow([model_index, test_avg_loss])

if __name__ == "__main__":
    main()
