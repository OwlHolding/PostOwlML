from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, Data2VecVisionModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
from transformers import RobertaTokenizer, Data2VecTextModel
from sentence_transformers import SentenceTransformer
from utils import get_label, get_id, mean_pooling, Identity
import requests


class PreprocessingDataset(Dataset):
    def __init__(self, news, content='Text', model=None, images_folder=None, image_preprocessor=None, tokenizer=None):
        super().__init__()
        self.news = news
        self.model = model
        self.images_folder = images_folder
        self.content = content
        self.image_preprocessor = image_preprocessor
        self.tokenizer = tokenizer
        if self.content == 'Text':
            self.text = self.tokenizer(
                self.news['Text'].tolist(),
                max_length=128,
                truncation=True,
                return_token_type_ids=False,
                padding='max_length',
                return_tensors='pt'
            )

    def __getitem__(self, idx):
        if self.content == 'Text':
            return self.text['input_ids'][idx], self.text['attention_mask'][idx]
        elif self.content == 'Image':
            if self.model is None:
                try:
                    return self.image_preprocessor(Image.open(self.images_folder / f'{self.news["NewsID"][idx]}.jpg'),
                                                   return_tensors="pt")['pixel_values']
                except Exception as e:
                    return self.image_preprocessor(
                        Image.open(self.images_folder / f'{self.news["NewsID"][idx]}.jpg').convert('RGB'),
                        return_tensors="pt")['pixel_values']
            else:
                return self.images_folder / f'{self.news["NewsID"][idx]}.jpg'
        else:
            raise ValueError

    def __len__(self):
        return len(self.news)

    @staticmethod
    def load_image(url_or_path):
        return Image.open(url_or_path)


def batch_collate_text(batch):
    input_ids, attention_mask = torch.utils.data._utils.collate.default_collate(batch)
    max_length = attention_mask.sum(dim=1).max().item()
    attention_mask, input_ids = attention_mask[:, :max_length], input_ids[:, :max_length]
    return input_ids, attention_mask


def batch_collate_image(batch):
    pixel_values = torch.utils.data._utils.collate.default_collate(batch)
    max_length = torch.ones(pixel_values.size()).sum(dim=1).max().item()
    pixel_values = pixel_values[:, :max_length]
    return pixel_values


def preparation_data(part='train'):
    users = pd.read_csv(f'datasets/MIND/{part}/behaviors.tsv', header=None, sep='\t')

    users.columns = ['index', 'UserID', 'Timestamp', 'News', 'News_marked']
    users.dropna(inplace=True)
    users.drop(columns=['Timestamp', 'index'], inplace=True)

    users['News_marked'] = users['News_marked'].apply(lambda x: x.split())
    users['News'] = users['News'].apply(lambda x: x.split())

    users['Labels'] = users['News_marked'].apply(get_label)
    users['News_marked'] = users['News_marked'].apply(get_id)

    news = pd.read_csv(f'datasets/MIND/{part}/news.tsv', header=None, sep='\t')
    news.columns = ['NewsID', "Category", "SubCategory", "Title", "Abstract", "URL", "Title Entities",
                    "Abstract Entities"]

    news['Abstract'].replace(np.NaN, '', inplace=True)
    news['Text'] = news['Abstract'] + '\n' + news['Title']

    news.drop(columns=['Category', 'SubCategory', 'Title', 'Abstract', 'URL',
                       'Title Entities', 'Abstract Entities'], inplace=True)

    users.reset_index(inplace=True)
    news.reset_index(inplace=True)
    users.to_feather(f'datasets/MIND/users_{part}.feather')
    news.to_feather(f'datasets/MIND/news_{part}.feather')
    return len(users['UserID'].unique()), len(news['NewsID'].unique())


def get_embedding_from_text(batch_size, path='datasets/MIND/news_train.feather'):
    news = pd.read_feather(path)
    text_encoder = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")
    print('\n' * 10)
    text_encoder.pooler.dense = Identity()
    dataset = PreprocessingDataset(
        news,
        content='Text',
        tokenizer=RobertaTokenizer.from_pretrained("facebook/data2vec-text-base")
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=batch_collate_text)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_encoder.to(device)
    embs = []
    with tqdm(total=len(loader)) as progress_bar:
        for batch in loader:
            input_ids, attention_mask = batch
            emb = text_encoder(input_ids=input_ids.to(device),
                               attention_mask=attention_mask.to(device)).last_hidden_state
            embs.append(mean_pooling(emb, attention_mask.to(device)).cpu().detach().numpy())
            progress_bar.update()
    embs = np.concatenate(embs, axis=0)
    res = []
    with tqdm(total=len(embs)) as progress_bar:
        for emb in embs:
            res.append(emb)
            progress_bar.update()
    news['ImageEmbedding'] = res
    news.to_feather(path)


def get_embedding_from_image(batch_size, images_folder, path='datasets/MIND/news_train.feather'):
    news = pd.read_feather(path)
    image_encoder = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base")
    dataset = PreprocessingDataset(
        news,
        content='Image',
        images_folder=images_folder,
        image_preprocessor=AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
    )

    loader = DataLoader(dataset, batch_size=batch_size)  # , collate_fn=batch_collate_image)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_encoder.to(device)
    embs = []
    print('\n' * 10)
    with tqdm(total=len(loader)) as progress_bar:
        for batch in loader:
            emb = image_encoder(pixel_values=batch.squeeze(1).to(device)).last_hidden_state
            embs.append(mean_pooling(emb, torch.ones(emb.size()[:-1], device=device)).cpu().detach().numpy())
            progress_bar.update()

    embs = np.concatenate(embs, axis=0)
    res = []
    with tqdm(total=len(embs)) as progress_bar:
        for emb in embs:
            res.append(emb)
            progress_bar.update()
    news['TextEmbedding'] = res
    news.to_feather(path)


def get_clip_embedding_from_text(batch_size, path='datasets/MIND/news_train.feather'):
    news = pd.read_feather(path)
    text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    print('\n' * 10)
    dataset = PreprocessingDataset(
        news,
        content='Text',
        tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=batch_collate_text)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_encoder.to(device)
    embs = []
    with tqdm(total=len(loader)) as progress_bar:
        for batch in loader:
            input_ids, attention_mask = batch
            emb = text_encoder({'input_ids': input_ids.to(device),
                                'attention_mask': attention_mask.to(device)})[
                'sentence_embedding'].cpu().detach().numpy()
            embs.append(emb)
            progress_bar.update()
    embs = np.concatenate(embs, axis=0)
    res = []
    with tqdm(total=len(embs)) as progress_bar:
        for emb in embs:
            res.append(emb)
            progress_bar.update()
    news['TextEmbedding'] = res
    news.to_feather(path)


def get_clip_embedding_from_image(images_folder, path='datasets/MIND/news_train.feather'):
    news = pd.read_feather(path)
    image_encoder = SentenceTransformer('clip-ViT-B-32')
    dataset = PreprocessingDataset(
        news,
        content='Image',
        images_folder=images_folder,
        model='clip',
        image_preprocessor=None
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_encoder.to(device)
    res = []
    embs = []
    print('\n' * 10)
    for i in range(len(dataset)):
        embs.append(dataset[i])
    for i in range(0, len(dataset), 128):
        for r in image_encoder.encode(list(map(dataset.load_image, embs[i:i + 128])),
                                      batch_size=128, device=device):
            res.append(r)
    print(len(res), len(news))
    news['ImageEmbedding'] = res
    news.to_feather(path)


if __name__ == '__main__':
    # with tqdm(total=2) as tq:
    #     tq.set_description('Processing the train part', refresh=True)
    #     users_len_train, news_len_train = preparation_data('train')
    #     tq.update()
    #     tq.set_description('Processing the validation part', refresh=True)
    #     users_len_val, news_len_val = preparation_data('val')
    #     tq.update()
    #     print(f'train:\n\tunique users {users_len_train}\n\tunique news {news_len_train}')
    #     print(f'validation:\n\tunique users {users_len_val}\n\tunique news {news_len_val}')
    get_clip_embedding_from_image(images_folder=Path('datasets/MIND/Images'), path=Path('datasets/MIND/news_val'
                                                                                        '.feather'))
    get_clip_embedding_from_text(10, path=Path('datasets/MIND/news_val.feather'))
