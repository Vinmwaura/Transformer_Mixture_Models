import torch
import torch.nn as nn
import torch.nn.functional as F

from .DecoderTransformer import DecoderTransformer

"""
Mixture of models, instead of mixture of experts.
"""
class Mixture_of_Models(nn.Module):
    def __init__(
            self,
            num_models,
            num_embeddings,
            embedding_dim,
            hidden_dim,
            capacity_factor=1.0,
            num_heads=8,
            num_blocks=6,
            out_classes=10,
            use_masked_attn=True,
            activation_type="gelu"):
        super().__init__()

        self.capacity_factor = capacity_factor

        self.models = nn.ModuleList()
        for _ in range(num_models):
            self.models.append(
                DecoderTransformer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_blocks=num_blocks,
                    out_classes=out_classes,
                    use_masked_attn=use_masked_attn,
                    activation_type=activation_type)
            )

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

    """
    Heuristic approach, forces models to be 'activated' a certain proportion for each input.
    """
    def compute_activated_mask(self, x_classifier, y_target):
        device = x_classifier.device.type

        N, K, Seq, Class = x_classifier.shape

        expert_capacity = (Seq // K) * self.capacity_factor
        expert_capacity = round(expert_capacity)

        x_classifier_flat = x_classifier.flatten(
            start_dim=0,
            end_dim=2)  # (N*K*Seq,Class)

        y_target = y_target.unsqueeze(dim=1)  # (N,1,Seq)
        y_target = y_target.repeat(1,K,1)  # (N,K,Seq)
        y_target_flat = y_target.flatten()  # (N*K*Seq,)

        # CrossEntropy Loss for each model.
        models_loss = F.cross_entropy(
            input=x_classifier_flat,
            target=y_target_flat,
            reduction='none')  # (N*K*Seq,)
        models_loss = models_loss.reshape(N,K,Seq)  # (N,K,Seq)

        # Get indices of loss sorted from smallest to largest.
        _, models_loss_indices = torch.sort(
            models_loss,
            dim=1,
            stable=True)  # (N,K,Seq)

        # Create Sequential chunks for models_loss_indices.
        models_loss_indices_chunks = torch.chunk(
            models_loss_indices,
            dim=2,
            chunks=Seq)  # Seq tuples of (N,K,1)
        
        # Count of tokens assigned to each model.
        token_count = torch.zeros(
            (N,K),
            device=device)  # (N,K)

        N_list = torch.arange(start=0, end=N, device=device)

        activated_models_indices_list = []

        # TODO: Optimize for-loop away, bottleneck here.
        for models_loss_indices_chunk in models_loss_indices_chunks:
            models_loss_indices_chunk = models_loss_indices_chunk.squeeze(dim=-1)  # (N,K)

            # Total count of tokens for each model.
            token_count_mask = torch.gather(
                token_count,
                dim=1,
                index=models_loss_indices_chunk)  # (N,K)

            # Test if model has reached capacity and if so route to next best model.
            token_count_mask = (token_count_mask < expert_capacity).float()  # (N,K)

            # Get best model that has not exceeded expert capacity.
            best_token_count_index = torch.argmax(
                token_count_mask,
                dim=1)  # (N,)

            # Model with lowest loss and hasn't maximized it's capacity.
            temp_activated_models = models_loss_indices_chunk[N_list, best_token_count_index]  # (N,)
            activated_models_indices_list.append(temp_activated_models)

            # Increment model token_count.
            token_count[N_list, temp_activated_models] += 1

        activated_models_indices = torch.stack(
            activated_models_indices_list,
            dim=1)  # (N,Seq)

        return activated_models_indices

    def forward(self, x_in, y_target=None):
        K = len(self.models)

        all_classifier = []
        all_confidence = []
        for model in self.models:
            x_classifier, x_confidence = model(x_in)

            all_classifier.append(x_classifier)
            all_confidence.append(x_confidence)

        combined_classifier = torch.stack(all_classifier, dim=1)  # (N,K,Seq,Class)
        combined_confidence = torch.stack(all_confidence, dim=1)  # (N,K,Seq,1)
        combined_confidence = combined_confidence.squeeze(dim=-1)  # (N,K,Seq)

        if y_target is not None:
            # Non-Differentiable calculation.
            combined_classifier_detached = combined_classifier.clone().detach()  # (N,K,Seq,Class)
            activated_models_indices = self.compute_activated_mask(
                x_classifier=combined_classifier_detached,
                y_target=y_target)

            activated_models_mask = F.one_hot(
                activated_models_indices,
                num_classes=K)  # (N,Seq,K)
            activated_models_mask = activated_models_mask.permute(0,2,1)  # (N,K,Seq)
            activated_models_mask = activated_models_mask.unsqueeze(dim=-1)  # (N,K,Seq,1)
        else:
            activated_models_indices = torch.argmax(
                combined_confidence,
                dim=1)  # (N,Seq)
            activated_models_mask = F.one_hot(
                activated_models_indices,
                num_classes=len(self.models))  # (N,Seq,K)
            activated_models_mask = activated_models_mask.permute(0,2,1)  # (N,K,Seq)
            activated_models_mask = activated_models_mask.unsqueeze(dim=-1)  # (N,K,Seq,1)

        # Differentiable calculation.
        out_combined_classifier = (combined_classifier * activated_models_mask)  # (N,K,Seq,Class)
        out_combined_classifier = out_combined_classifier.sum(dim=1)  # (N,Seq,Class)
        
        return out_combined_classifier, combined_confidence, activated_models_indices
