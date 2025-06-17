import os
import csv
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

from utils.model_utils import (
    save_model,
    load_model)

def checkpoint_model(
        data_dict,
        out_dir,
        model,
        model_optim,
        logging):
    global_steps = data_dict["global_steps"]

    # Save model that has achieved max TPR with the dataset.
    model_dict = {
        **data_dict,
        "model": model.state_dict(),
        "optimizer": model_optim.state_dict()}

    save_status = save_model(
        model_dict=model_dict,
        dest_path=out_dir,
        init_folder=True,
        file_name=f"{global_steps}_model.pt",
        logging=logging)
    if save_status is True:
        logging("Successfully saved model.")
    else:
        logging("Error occured saving model.")

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.001:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "Mixtures of [Blocks or Models]"

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
        "--tr-dataset-path",
        help="File path to training csv dataset file.",
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
        "--use-activation-checkpoint",
        help="Use activation checkpointing to trade computation speed for more memory.",
        type=bool,
        default=False)
    parser.add_argument(
        "--checkpoint-steps",
        help="Steps for checkpointing and/or testing model.",
        type=int,
        default=1_000)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint to load from (if any).",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "--lr-steps",
        help="# Global steps in between halving learning rate..",
        default=50_000,
        type=int)
    parser.add_argument(
        "--load-optim",
        action='store_true',
        help="Load model's optimizer's weights and parameters, if loading model.")
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
    lr_steps = args["lr_steps"]  # Global steps in between halving learning rate.
    load_optim = args["load_optim"]  # Reload saved optimizer weights.
    vocab_dataset_path = args["vocab_dataset_path"]  # Vocabulary json file path (*.json).
    tr_dataset_path = args["tr_dataset_path"]  # Training csv file path (*.csv).
    tst_dataset_path = args["tst_dataset_path"]  # Training csv file path (*.csv).
    batch_size = args["batch_size"]  # Batch size of training dataset.
    model_checkpoint = args["model_checkpoint"]  # Filepath to models saved.
    use_activation_checkpoint = args["use_activation_checkpoint"]
    checkpoint_steps = args["checkpoint_steps"]  # Steps to checkpoint model.
    out_dir = args["out_dir"]  # Destination path for model.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    config_json = args["config_path"]  # Load and Parse config JSON.
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    # Training params.
    global_steps = 0
    max_global_steps = config_dict["max_global_steps"]

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
    tr_dataset = SubWord_Dataset(
        csv_fpath=tr_dataset_path,
        start_token=start_token,
        padding_token=padding_token,
        context_window=context_window)
    tst_dataset = SubWord_Dataset(
        csv_fpath=tst_dataset_path,
        start_token=start_token,
        padding_token=padding_token,
        context_window=context_window)

    # Dataloaders.
    tr_dataloader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True)
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True)
    tst_iterator = iter(tst_dataloader)

    if mixture_type == "models":
        mixture_model = MixtureofModels(
            num_models=num_mixture,
            num_embeddings=vocab_size + 2,  # Includes [START] and [PAD] tokens.
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            out_classes=vocab_size + 1,  # Includes only [PAD] tokens.
            use_checkpoint=use_activation_checkpoint,
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
            use_checkpoint=use_activation_checkpoint,
            activation_type=activation_type)
    else:
        raise Exception("Invalid Mixture type!")

    # Load Transformer Model checkpoints if any.
    if model_checkpoint is not None:
        logging.info("Loading Model...")
        classifier_status, classifier_dict = load_model(model_checkpoint)
        if not classifier_status:
            raise Exception("An error occured while loading model checkpoint!")

        mixture_model.custom_load_state_dict(classifier_dict["model"])
        mixture_model = mixture_model.to(device)

        mixture_model_optim = torch.optim.Adam(
            mixture_model.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

        # Load Optimizer params and global steps params.
        if load_optim:
            logging.info("Resuming Training using saved optimizer weights and global_steps...")
            mixture_model_optim.load_state_dict(classifier_dict["optimizer"])
    else:
        mixture_model = mixture_model.to(device)

        mixture_model_optim = torch.optim.Adam(
            mixture_model.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

    # Model Params size.
    model_params_size = sum(param.numel() for param in mixture_model.parameters())

    # https://pytorch.org/docs/stable/amp.html
    scaler = torch.cuda.amp.GradScaler()

    # Log file path.
    log_path = os.path.join(
        out_dir,
        f"{project_name}.log")

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

    logging.info(f"{project_name}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("#" * 100)
    logging.info("Dataset Parameters.")
    logging.info(f"Vocab Size: {vocab_size:,}")
    logging.info(f"Total Train Dataset: {len(tr_dataset):,}")
    logging.info(f"Context window: {context_window:,}")
    logging.info(f"Train Batch Size: {batch_size:,}")
    logging.info("#" * 100)
    logging.info("Model Parameters.")
    logging.info(f"Total Model Param size: {model_params_size:,}")
    logging.info(f"Mixture type: {mixture_type}")
    logging.info(f"Number of mixtures: {num_mixture:,}")
    logging.info(f"Number of heads: {num_heads:,}")
    logging.info(f"Number of blocks: {num_blocks:,}")
    logging.info(f"Embedding Dimension: {embedding_dim:,}")
    logging.info(f"Activation Type: {activation_type}")
    logging.info(f"Model Learning Rate: {mixture_model_optim.param_groups[0]['lr']:,}")
    logging.info("#" * 100)
    logging.info("Training Parameters.")
    logging.info(f"Step: {global_steps:,}")
    logging.info(f"Max Global step: {max_global_steps:,}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps:,}")
    logging.info("#" * 100)

    model_data_dict = {
        "vocab": vocab,
        "mixture_type": mixture_type,
        "num_mixture": num_mixture,
        "num_heads": num_heads,
        "num_blocks": num_blocks,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "context_window": context_window,
        "activation_type": activation_type,
        "global_steps": global_steps}

    # Training starts here.
    stop_training = False
    while not stop_training:
        # TODO: Remove third value from data.
        for index, (in_seq, target_seq) in enumerate(tr_dataloader):
            # Checkpoint and test model.
            if global_steps % checkpoint_steps == 0:
                model_data_dict["global_steps"] = global_steps
                checkpoint_model(
                    data_dict=model_data_dict,
                    out_dir=out_dir,
                    model=mixture_model,
                    model_optim=mixture_model_optim,
                    logging=logging.info)

            # Training Data.
            in_seq = in_seq.to(device)  # (N,Seq)
            target_seq = target_seq.to(device)  # (N,Seq)

            mixture_model.train(mode=True)

            """
            Train Classifier.
            """
            # Runs the forward pass under ``autocast``.
            with torch.autocast(device_type=device, dtype=torch.float16):
                out_classifier = mixture_model(in_seq)

                target_seq_flat = target_seq.flatten()  # (N*Seq,)
                out_classifier_flat = out_classifier.flatten(
                    start_dim=0,
                    end_dim=1)  # (N*Seq,Class)

                tr_classifier_loss = F.cross_entropy(
                    out_classifier_flat,
                    target_seq_flat,
                    ignore_index=len(vocab))
                if torch.isnan(tr_classifier_loss):
                    raise Exception("NaN encountered during training.")
                
            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(tr_classifier_loss).backward()

            scaler.step(mixture_model_optim)

            # Updates the scale for next iteration.
            scaler.update()

            mixture_model_optim.zero_grad()

            correct_predictions = torch.eq(
                torch.argmax(out_classifier_flat, dim=1),
                target_seq_flat
            ).long().sum().item()

            train_classifier_loss = tr_classifier_loss.item()

            """
            Test Classifier.
            """
            try:
                tst_in_seq, tst_target_seq = next(tst_iterator)
            except StopIteration:
                tst_iterator = iter(tst_dataloader)
                tst_in_seq, tst_target_seq = next(tst_iterator)

            tst_in_seq = tst_in_seq.to(device)
            tst_target_seq = tst_target_seq.to(device)

            mixture_model.eval()

            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
                tst_out_classifier = mixture_model(tst_in_seq)

                tst_target_seq_flat = tst_target_seq.flatten()  # (N*Seq,)
                tst_out_classifier_flat = tst_out_classifier.flatten(
                    start_dim=0,
                    end_dim=1)  # (N*Seq,Class)

                tst_classifier_loss = F.cross_entropy(
                    tst_out_classifier_flat,
                    tst_target_seq_flat,
                    ignore_index=len(vocab))

            test_classifier_loss = tst_classifier_loss.item()

            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | Train Classifier Loss: {:,.5f} | Test Classifier Loss: {:,.5f} | Correct: {:,} | Total: {:,} | LR: {:.3E}".format(
                global_steps + 1,
                index + 1,
                len(tr_dataloader),
                train_classifier_loss,
                test_classifier_loss,
                correct_predictions,
                in_seq.numel(),
                mixture_model_optim.param_groups[0]['lr'])

            logging.info(message)

            global_steps = global_steps + 1

            # Stop training when stopping criteria is met.
            if global_steps >= max_global_steps:
                stop_training = True
                break

            if global_steps % lr_steps==0 and global_steps > 0:
                # Update LR.
                for mixture_model_optim_ in mixture_model_optim.param_groups:
                    mixture_model_optim_['lr'] = mixture_model_optim_['lr'] * 0.5

        model_data_dict["global_steps"] = global_steps
        checkpoint_model(
            data_dict=model_data_dict,
            out_dir=out_dir,
            model=mixture_model,
            model_optim=mixture_model_optim,
            logging=logging.info)

if __name__ == "__main__":
    main()
