import torch

def token_level_accuracy(labels: torch.LongTensor, pad_token: int, pred_logits: torch.FloatTensor, reduction = 'mean', tokenizer = None):
    mask = labels.ne(pad_token)
    n_correct = torch.sum(
        pred_logits.argmax(-1).masked_select(mask).eq(labels.masked_select(mask))
    )
    total = torch.sum(mask)
    accuracy = n_correct / total
    return n_correct, total, accuracy

def token_level_accuracy_v2(labels: torch.LongTensor, pad_token: int, pred_logits: torch.FloatTensor, reduction = 'mean'):
    shift_logits = pred_logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    mask = shift_labels.ne(pad_token)
    n_correct = torch.sum(
        shift_logits.argmax(-1).masked_select(mask).eq(shift_labels.masked_select(mask))
    )
    total = torch.sum(mask)
    accuracy = n_correct / total
    return n_correct, total, accuracy


