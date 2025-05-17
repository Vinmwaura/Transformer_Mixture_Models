import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    LinearBlock,
    TransformerBlock,
    PositionalEncoding)


"""
Transformer Architecture.
Decoder-only Transformer model.
"""
class DecoderTransformer(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            hidden_dim,
            num_heads=8,
            num_blocks=6,
            out_classes=8,
            use_classifier_block=True,
            activation_type="gelu"):
        super().__init__()

        self.use_classifier_block = use_classifier_block

        # Learnable Embedding and Positional Encoding.
        self.emb_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim)
        self.pos_layer = PositionalEncoding()

        # Decoder Blocks.
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_blocks.append(
                TransformerBlock(
                    heads=num_heads,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    is_causal=True,
                    activation_type=activation_type))

        if self.use_classifier_block:
            # Classifier Block.
            self.classifier_block = nn.Sequential(
                LinearBlock(
                    in_dim=embedding_dim,
                    out_dim=hidden_dim,
                    use_activation=True),
                LinearBlock(
                    in_dim=hidden_dim,
                    out_dim=out_classes,
                    use_activation=False))

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
        # Embedding Layer + Positional Encoding.
        x_emb = self.emb_layer(x)
        x_emb_pos = self.pos_layer(x_emb)

        # Classifier Blocks.
        x_decoder = 1 * x_emb_pos
        for decoder_block in self.decoder_blocks:
            x_decoder = decoder_block(x_decoder)

        if self.use_classifier_block:
            x_classifier = self.classifier_block(x_decoder)
            return x_classifier
        else:
            return x_decoder
