import torch
import random
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

def get_label(news):
    return list(map(lambda x: x.split('-')[1], news))


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
        self.epoch = 0
        self.users = users
        self.images_folder = images_folder
        self.images_rate = images_rate

    def __getitem__(self, idx):
        if idx == 0:
            self.epoch += 1
        row = self.users.iloc[idx]
        news_u = []
        for news_id in row['News'][:self.epoch]:
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
        return news_u, news_i, torch.LongTensor(row['Labels'])

    def __len__(self):
        return len(self.users)

    def processing_image(self, id_):
        return self.image_preprocessor(Image.open(self.images_folder/id_), return_tensors="pt")

    def processing_text(self, id_):
        return self.tokenizer(
            self.news[self.news['NewsID'] == id_]['Text'],
            max_length=128,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_tensors='pt'
        )


