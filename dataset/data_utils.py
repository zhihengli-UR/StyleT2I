import torch
import random
import numpy as np


from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pad_text_seq_collate(dict_lst):
    text_batch = []
    text_len = []

    has_word_embeds = "word_embeds" in dict_lst[0]

    for data_dict in dict_lst:
        if "word_embeds" in data_dict:
            word_embeds = data_dict["word_embeds"]
            text_batch.append(word_embeds)
            text_len.append(len(word_embeds))
            del data_dict["word_embeds"]

    batched_dict = default_collate(dict_lst)

    if has_word_embeds:
        text_len = torch.tensor(text_len, dtype=torch.long)
        sorted_text_len, sort_text_len_ind = torch.sort(
            text_len, dim=0, descending=True
        )
    else:
        sort_text_len_ind = torch.arange(len(dict_lst), dtype=torch.long)

    new_batched_dict = {}
    for key, value in batched_dict.items():
        if key == "filename":
            new_value = [None] * len(value)
            for filename, sort_idx in zip(value, sort_text_len_ind.tolist()):
                new_value[sort_idx] = filename
            new_batched_dict[key] = new_value
        else:
            new_batched_dict[key] = value[sort_text_len_ind]
    batched_dict = new_batched_dict

    if has_word_embeds:
        text_batch = pad_sequence(text_batch, batch_first=True, padding_value=0)[
            sort_text_len_ind
        ]
        batched_dict["word_embeds"] = text_batch
        batched_dict["text_len"] = sorted_text_len

    return batched_dict
