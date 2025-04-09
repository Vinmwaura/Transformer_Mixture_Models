import torch
import torch.nn.functional as F

def generate_text(
        vocab,
        device,
        num_models,
        start_token,
        mixture_model,
        context_window,
        logging=print,
        temperature=1.0):
    mixture_model.eval()

    generated_token_list = [start_token]
    generated_token_tensor = torch.tensor(
        [generated_token_list],
        device=device)  # (1,1)

    for _ in range(context_window):
        with torch.no_grad():
            out_classifier, _, activated_indices = mixture_model(
                generated_token_tensor,
                y_target=None)  # (1,Seq,Class)

        probs = F.softmax(
            out_classifier / temperature,
            dim=2)  # (1,Seq,Class)

        probs_token = probs[:,-1,:]  # (1,Class)

        # Pick most likely token for next generation for each Token Sequence (N*Seq,).
        next_token = torch.multinomial(probs_token, 1)  # (1,1)

        # Save last token for next prediction.
        generated_token_tensor = torch.cat(
            (generated_token_tensor, next_token),
            dim=1)  # (N,Seq+1)

    activated_mask = F.one_hot(
        activated_indices,
        num_classes=num_models)  # (N,Seq,K)

    activated_models_K_sum = activated_mask.sum(dim=(0,1))  # (K,)
    activated_models_total_sum = activated_mask.sum()  # (1,)
    activated_models_percent = activated_models_K_sum / activated_models_total_sum

    vocab_size = len(vocab)

    generated_token_tensor = generated_token_tensor.squeeze(dim=0)  # (Seq,)
    generated_token_list = generated_token_tensor[1:].tolist()

    # Remove invalid tokens if any like padding token, not in vocab list.
    cleaned_pred_tokens = [clean_token for clean_token in generated_token_list if clean_token < vocab_size]
    pred_token_list = [vocab[c] for c in cleaned_pred_tokens]
    pred_txt = "".join(pred_token_list)

    logging(f"Activated model (%): {activated_models_percent.tolist()}")
    logging(f"Generated text:\n{pred_txt}")
    logging("#" * 100)
