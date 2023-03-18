import core.telegram as telegram
import asyncio
from starlette.testclient import TestClient
from app import app
from core.files import save_config, load_config
import json
import random
import shutil


client = TestClient(app)
posts = dict()
markup = dict()


def test_telegram_channels():
    assert asyncio.run(telegram.get_posts("forbesrussia", 10, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("mintsifry", 10, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("SKOLKOVO_Start", 10, 0))[1] == 200


def test_telegram_count():
    assert asyncio.run(telegram.get_posts("forbesrussia", 100, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("forbesrussia", 500, 0))[1] == 200


def test_ping():
    response = client.post('/register/1')
    assert response.status_code // 10 == 20


def test_register_channel():
    response = client.post('/regchannel/1/forbesrussia')
    assert response.status_code // 10 == 20
    posts['forbesrussia'] = response.json()
    response = client.post('/regchannel/1/gradientdip')
    assert response.status_code // 10 == 20
    posts['gradientdip'] = response.json()


def test_train():
    data = {
        'posts': posts['forbesrussia']['posts'],
        'labels': [random.choice([0, 1]) for _ in range(len(posts['forbesrussia']['posts']))],
        'finetune': False
    }
    response = client.post('/train/1/forbesrussia', data=json.dumps(data))
    assert response.status_code // 10 == 20
    data = {
        'posts': posts['gradientdip']['posts'],
        'labels': [random.choice([0, 1]) for _ in range(len(posts['gradientdip']['posts']))],
        'finetune': False
    }
    response = client.post('/train/1/gradientdip', data=json.dumps(data))
    assert response.status_code // 10 == 20


def test_predict():
    response = client.post('/predict/1/forbesrussia', data=json.dumps({'time': 0}))
    assert response.status_code // 10 == 20
    markup['forbesrussia'] = [response.json()['markup']]
    assert len(response.json()['posts']) == 5
    response = client.post('/predict/1/gradientdip', data=json.dumps({'time': 0}))
    assert response.status_code // 10 == 20
    markup['gradientdip'] = [response.json()['markup']]
    assert len(response.json()['posts']) == 5


def test_owl_learnig_step():
    for _ in range(6):
        data = {
            'posts': markup['forbesrussia'],
            'labels': [random.choice([0, 1])],
            'finetune': True
        }
        response = client.post('/train/1/forbesrussia', data=json.dumps(data))
        assert response.status_code // 10 == 20
        response = client.post('/predict/1/forbesrussia', data=json.dumps({'time': 0}))
        assert response.status_code // 10 == 20
        markup['forbesrussia'] = [response.json()['markup']]

    for _ in range(6):
        data = {
            'posts': markup['gradientdip'],
            'labels': [random.choice([0, 1])],
            'finetune': True
        }
        response = client.post('/train/1/gradientdip', data=json.dumps(data))
        assert response.status_code // 10 == 20
        response = client.post('/predict/1/gradientdip', data=json.dumps({'time': 0}))
        assert response.status_code // 10 == 20
        markup['gradientdip'] = [response.json()['markup']]


def test_catboost():
    config = load_config(1, 'forbesrussia')
    config['model'] = 'CatBoost'
    save_config(1, 'forbesrussia', config['model'])
    config = load_config(1, 'gradientdip')
    config['model'] = 'CatBoost'
    save_config(1, 'gradientdip', config['model'])
    assert True


def test_owl_learnig_step_cb():
    for _ in range(6):
        data = {
            'posts': markup['forbesrussia'],
            'labels': [random.choice([0, 1])],
            'finetune': True
        }
        response = client.post('/train/1/forbesrussia', data=json.dumps(data))
        assert response.status_code // 10 == 20
        response = client.post('/predict/1/forbesrussia', data=json.dumps({'time': 0}))
        assert response.status_code // 10 == 20
        markup['forbesrussia'] = [response.json()['markup']]

    for _ in range(7):
        data = {
            'posts': markup['gradientdip'],
            'labels': [random.choice([0, 1])],
            'finetune': True
        }
        response = client.post('/train/1/gradientdip', data=json.dumps(data))
        assert response.status_code // 10 == 20
        response = client.post('/predict/1/gradientdip', data=json.dumps({'time': 0}))
        assert response.status_code // 10 == 20
        markup['gradientdip'] = [response.json()['markup']]


def test_predict_cb():
    response = client.post('/predict/1/forbesrussia', data=json.dumps({'time': 0}))
    assert response.status_code // 10 == 20
    markup['forbesrussia'] = [response.json()['markup']]
    assert len(response.json()['posts']) == 5
    response = client.post('/predict/1/gradientdip', data=json.dumps({'time': 0}))
    assert response.status_code // 10 == 20
    markup['gradientdip'] = [response.json()['markup']]
    assert len(response.json()['posts']) == 5


def test_remove_dir():
    shutil.rmtree('users\\1')
    assert True
