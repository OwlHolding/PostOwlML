import json
import time
import shutil

from starlette.testclient import TestClient

import core.telegram as telegram
from app import app
from files import save_config, load_config, load_dataset, save_dataset

client = TestClient(app)
posts = dict()
markup = dict()
channels = ['forbesrussia',
            'postupashki',
            'airi_research_institute',
            'mintsifry',
            'bloomberg_ru',
            'RussianHackers_Channel',
            'BDataScienceM',
            'gradientdip',
            'yandex',
            'nn_for_science']


def test_telegram_channels():
    for channel in channels:
        assert telegram.get_posts(channel, 10, 0)[1] == 200


def test_telegram_channels_rss():
    for channel in channels:
        assert telegram.get_posts_rss(channel, 10, 0)[1] == 200


def test_telegram_count():
    assert telegram.get_posts(channels[0], 100, 0)[1] == 200
    assert telegram.get_posts(channels[0], 500, 0)[1] == 200


def test_telegram_count_rss():
    assert telegram.get_posts_rss(channels[0], 100, 0)[1] == 200
    assert telegram.get_posts_rss(channels[0], 500, 0)[1] == 200


def test_ping():
    response = client.post('/register/1')
    assert response.status_code // 10 == 20


def test_register_channel():
    for channel in channels:
        response = client.post(f'/regchannel/1/{channel}')
        assert response.status_code // 10 == 20
        posts[channel] = response.json()


def test_wrong_channel():
    response = client.post('/regchannel/1/dhfhsdhfkhkffsdsdsdjk')
    assert response.status_code == 400


def test_wrong_user():
    response = client.post(f'/regchannel/2/{channels[0]}')
    assert response.status_code == 404


def test_train():
    for channel in channels:
        data = json.dumps({
            'posts': posts[channel]['posts'],
            'labels': [i % 2 for i in range(len(posts[channel]['posts']))],
            'finetune': False
        })
        response = client.post(f'/train/1/{channel}', data=data)
        assert response.status_code // 10 == 20


def test_predict():
    for channel in channels:
        response = client.post(f'/predict/1/{channel}', data=json.dumps({'time': 0}))
        markup[channel] = [response.json()['markup']]
        assert response.status_code // 10 == 20
        assert isinstance(response.json()['markup'], str)
        assert isinstance(response.json()['posts'], list)


def get_finetune_data(channel):
    return json.dumps({
        'posts': markup[channel],
        'labels': [i % 2 for i in range(len(markup[channel]))],
        'finetune': True
    })


def test_owl_learning_step():
    for channel in channels:
        for _ in range(6):
            response = client.post(f'/train/1/{channel}', data=get_finetune_data(channel))
            assert response.status_code // 10 == 20
            response = client.post(f'/predict/1/{channel}', data=json.dumps({'time': 0}))
            assert response.status_code // 10 == 20
            markup[channel] = [response.json()['markup']]


def test_owl_learning_step_cb():
    for channel in channels:
        config = load_config(1, channel)
        config['model'] = 'CatBoost'
        save_config(1, channel, config)

        dataset = load_dataset(1, channel)
        j = 0
        for i in dataset[dataset['labels'].isna()].index:
            if (dataset['labels'].notna().sum() - 10) % 6 == 0:
                break
            j += 1
            dataset.at[i, 'labels'] = j % 2
        save_dataset(user_id=1, channel=channel, dataset=dataset)
        data = {
            'posts': posts[channel]['posts'][:8],
            'labels': [i % 2 for i in range(len(posts[channel]['posts']))][:8],
            'finetune': True
        }
        response = client.post(f'/train/1/{channel}', data=json.dumps(data))
        assert response.status_code // 10 == 20


def test_predict_cb():
    for channel in channels:
        response = client.post(f'/predict/1/{channel}', data=json.dumps({'time': 0}))
        assert response.status_code // 10 == 20
        assert isinstance(response.json()['markup'], str)
        assert isinstance(response.json()['posts'], list)


def test_remove_dir():
    time.sleep(5)
    shutil.rmtree('users/1')
    assert True