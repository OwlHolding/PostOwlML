import os
import random

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_label(news):
    return list(map(lambda x: int(x.split('-')[1]), news))


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_id(news):
    return list(map(lambda x: x.split('-')[0], news))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MultiModalDataset(Dataset):
    def __init__(self, news, users, stage, images_rate=0.5):
        super().__init__()
        self.news = news
        self.news['ImageEmbedding'] = self.news['ImageEmbedding'].apply(lambda x: x.astype(np.float32))
        self.news['TextEmbedding'] = self.news['ImageEmbedding'].apply(lambda x: x.astype(np.float32))
        self.news.set_index('NewsID', inplace=True)
        self.max_len = min(max(users['News'].apply(len)), max(users['News_marked'].apply(len)))
        self.stage = stage if stage < self.max_len else self.max_len

        self.users = users[users['News'].apply(len) >= self.stage][users['News_marked'].apply(len) >= self.stage]
        self.images_rate = images_rate

    def __getitem__(self, idx):
        row = self.users.iloc[idx]
        return list(map(self.get_embedding, row['News'][:self.stage])), list(map(self.get_embedding, row['News_marked'][:self.stage])), row['Labels'][:self.stage].astype(float)

    def get_embedding(self, id_):
        if random.random() > self.images_rate:
            return self.news['ImageEmbedding'][id_]
        return self.news['TextEmbedding'][id_]

    def __len__(self):
        return len(self.users)

