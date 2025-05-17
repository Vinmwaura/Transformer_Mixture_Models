import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    LinearBlock,
    TransformerBlock,
    PositionalEncoding)

from .DecoderTransformer import DecoderTransformer

from torch.utils.checkpoint import checkpoint

"""
Mixture of Models.
Split output layer such that each model predicts a part of the output.
"""
class MixtureofModels(nn.Module):
    def __init__(
            self,
            num_models,
            num_embeddings,
            embedding_dim,
            hidden_dim,
            num_heads=8,
            num_blocks=6,
            out_classes=8,
            use_checkpoint=False,
            activation_type="gelu"):
        super().__init__()

        # Activation checkpoint: trades compute for memory.
        self.use_checkpoint = use_checkpoint

        # Equally split classes 'num_models' times.
        split_classes = self.split_classes(
            class_size=out_classes,
            num_models=num_models)

        # Mixture of Models.
        # Each of the models should ideally be distributed on other devices.
        self.mixture_of_models = nn.ModuleList()
        for model_index in range(num_models):
            self.mixture_of_models.append(
                DecoderTransformer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_blocks=num_blocks,
                    out_classes=split_classes[model_index],
                    use_classifier_block=True,
                    activation_type=activation_type))

    # Splits classes equally to fit into each mixture of models.
    def split_classes(self, class_size, num_models):
        base = class_size // num_models
        remainder = class_size % num_models
        return [base + 1 if i < remainder else base for i in range(num_models)]

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"No Layer found: {name}, skipping")
                continue
            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                print(f"Skipped: {name}")
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # Backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x):
        # Mixture of Models.
        x_mom = []
        for model in self.mixture_of_models:
            if self.use_checkpoint:
                x_model = checkpoint(
                    model,
                    x,
                    use_reentrant=False)
            else:
                x_model = model(x)  # (N,Seq,Class_chunks)

            x_mom.append(x_model)

        x_stacked = torch.cat(x_mom, dim=-1) # (N,Seq,Class)

        return x_stacked
