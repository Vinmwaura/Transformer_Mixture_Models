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
Mixture of Blocks.
Block being a groupings of multiple layers or blocks
e.g (Multi-Head Attn. + Add & Norm + FeedForward + Add & Norm) layers
can be seen as one block.
"""
class MixtureofBlocks(nn.Module):
    def __init__(
            self,
            num_mixture,
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

        # Equally split dimension 'num_mixture' times.
        split_dims = self.split_dim(
            dim_len=embedding_dim,
            num_mixture=num_mixture)

        # Mixture of Transformer Blocks.
        # Each of the blocks should ideally be distributed on other devices.
        self.mixture_of_blocks = nn.ModuleList()
        for mixture_index in range(num_mixture):
            self.mixture_of_blocks.append(
                nn.Sequential(
                    DecoderTransformer(
                        num_embeddings=num_embeddings,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        num_blocks=num_blocks,
                        out_classes=out_classes,
                        use_classifier_block=False,
                        activation_type=activation_type),
                    LinearBlock(
                        in_dim=embedding_dim,
                        out_dim=split_dims[mixture_index],
                        use_activation=True,
                        activation_type=activation_type)
                ))

        # Classifier Block.
        # Aggregate the embeddings from other mixtures and holistically process them.
        self.classifier_block = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=out_classes,
                use_activation=False))

    # Splits dimensions equally to fit into each mixture of blocks.
    def split_dim(self, dim_len, num_mixture):
        base = dim_len // num_mixture
        remainder = dim_len % num_mixture
        return [base + 1 if i < remainder else base for i in range(num_mixture)]

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
        # Mixture of Blocks.
        x_mob = []
        for model_block in self.mixture_of_blocks:
            if self.use_checkpoint:
                x_block = checkpoint(
                    model_block,
                    x,
                    use_reentrant=False)
            else:
                x_block = model_block(x)  # (N,Seq,Dim_chunks)

            x_mob.append(x_block)

        x_stacked = torch.cat(x_mob, dim=-1) # (N,Seq,Dim)
        x_classifier = self.classifier_block(x_stacked)

        return x_classifier
