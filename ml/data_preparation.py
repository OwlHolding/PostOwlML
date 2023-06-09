from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import get_label, get_id


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

if __name__ == '__main__':
    with tqdm(total=2) as tq:
        tq.set_description('Processing the train part', refresh=True)
        users_len_train, news_len_train = preparation_data('train')
        tq.update()
        tq.set_description('Processing the validation part', refresh=True)
        users_len_val, news_len_val = preparation_data('val')
        tq.update()
        print(f'train:\n\tunique users {users_len_train}\n\tunique news {news_len_train}')
        print(f'validation:\n\tunique users {users_len_val}\n\tunique news {news_len_val}')
