import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    LinearBlock,
    TransformerBlock,
    PositionalEncoding)

"""
Transformer Architecture.
Decoder-only Transformer models.
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
            use_masked_attn=True,
            activation_type="gelu"):
        super().__init__()

        # Learnable Embedding and Positional Encoding.
        self.emb_layers = nn.Sequential(
            nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim),
            PositionalEncoding())

        # Decoder Blocks.
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_blocks.append(
                TransformerBlock(
                    heads=num_heads,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    use_self_attn=True,
                    use_cross_attn=False,
                    use_masked_attn=use_masked_attn,
                    activation_type=activation_type
                )
            )

        self.classifier = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim//2,
                out_dim=hidden_dim,
                use_activation=True),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=out_classes,
                use_activation=False))

        self.confidence = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim//2,
                out_dim=hidden_dim,
                use_activation=True),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=1,
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
        x_dec = self.emb_layers(x)

        # Decoder Blocks.
        for decoder_block in self.decoder_blocks:
            x_dec = decoder_block(x_dec)

        N, Seq, D = x_dec.shape
        D_split = D // 2

        x_dec_split = x_dec.reshape(
            N, Seq, 2, D_split).permute(2,0,1,3)  # (2,N,Seq,D/2)

        x_class = self.classifier(x_dec_split[0])
        x_confidence = self.confidence(x_dec_split[1])

        return x_class, x_confidence
