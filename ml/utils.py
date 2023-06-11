import torch
import random
import os
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_label(news):
    return list(map(lambda x: int(x.split('-')[1]), news))


def get_id(news):
    return list(map(lambda x: x.split('-')[0], news))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MultiModalDataset(Dataset):
    def __init__(self, news, users, images_folder, image_preprocessor, tokenizer, images_rate=0.5, ):
        super().__init__()
        self.image_preprocessor = image_preprocessor
        self.tokenizer = tokenizer
        self.news = news
        self.news.set_index('NewsID', inplace=True)

        self.users = users
        self.images_folder = images_folder
        self.images_rate = images_rate
        self.max_len = max(users['News'].apply(len))
        self.stage = 1

    def __getitem__(self, idx):
        row = self.users.iloc[idx]
        news_u = []
        if self.stage < self.max_len:
            if idx % 1000 == 999:
                self.stage += 1
        for news_id in row['News'][:self.stage]:
            if random.random() < self.images_rate:
                news_u.append({'x': self.processing_image(news_id), 'content_type': 'image'})
            else:
                news_u.append({'x': self.processing_text(news_id), 'content_type': 'text'})
        news_i = []
        for news_id in row['News_marked']:
            if random.random() < self.images_rate:
                news_i.append({'x': self.processing_image(news_id), 'content_type': 'image'})
            else:
                news_i.append({'x': self.processing_text(news_id), 'content_type': 'text'})

        return news_u, news_i, torch.FloatTensor(row['Labels'])

    def __len__(self):
        return len(self.users)

    def processing_image(self, id_):
        return self.image_preprocessor(Image.open(self.images_folder / f'{id_}.jpg'), return_tensors="pt")

    def processing_text(self, id_):
        return self.tokenizer(
            self.news.loc[id_]['Text'],
            max_length=128,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_tensors='pt'
        )
