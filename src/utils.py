import torch
from torch import tensor
from torch.utils.data import DataLoader

def pad_sent(sents, max_seq_len):
    max_seq_len = min(100, max_seq_len)
    out = []
    lens = []
    for i in range(len(sents)):
        sent = sents[i]
        append_len = min(len(sent), max_seq_len)
        lens.append(append_len)
        append_tensor = tensor([sent[j] if j < len(sent) else 0 for j in range(max_seq_len)]).unsqueeze(0)
        out.append(append_tensor)
    out = torch.cat(out)
    return out, lens


def collate_fn(sample):

    labels = tensor([s[1] for s in sample])
    sents = [s[0] for s in sample]
    max_seq_len = max([sent.shape[0] for sent in sents])
    sents, lens = pad_sent(sents, max_seq_len)
    return sents, labels, lens

def generate_val_set(dataset, collate_fn=collate_fn):
    _, val_set = enumerate(DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)).__next__()
    return val_set
