"""Модуль управления файлами для пар пользователь:канал"""
import os
import multiprocessing
import pickle
import pandas as pd
import json


class RLookPool:

    def __init__(self):
        self.values = {}

    def __getitem__(self, item):
        if not (item in self.values.keys()):
            self.values[item] = multiprocessing.RLock()

        return self.values[item]


locks = RLookPool()


def register_channel(user_id: [int, str], channel: str) -> bool:
    """Добавляет канал существующему пользователю"""
    with locks[user_id]:
        if not os.path.exists(f'users/{user_id}/{channel}'):
            os.makedirs(f'users/{user_id}/{channel}')
            with open(f'users/{user_id}/{channel}/model.pk', 'w') as f:
                pass
            with open(f'users/{user_id}/{channel}/tfidf.pk', 'w') as f:
                pass
            dataset = pd.DataFrame(columns=['posts', 'labels', 'confidence']).reset_index()
            dataset.to_feather(f'users/{user_id}/{channel}/dataset.feather')
            config = {
                "model": "SVM",
            }
            with open(f'users/{user_id}/{channel}/config.json', 'w') as f:
                json.dump(config, f)

            return False

        return True


def register_user(user_id: [int, str]) -> bool:
    """Регистрирует нового пользователя в системе"""
    with locks[user_id]:
        if not os.path.exists(f'users/{user_id}'):
            os.makedirs(f'users/{user_id}')
            return False

        return True


def save_model(user_id: [int, str], channel: str, model, tfidf) -> None:
    """Сохраняет модель и tfidf"""
    with locks[user_id]:
        with open(f'users/{user_id}/{channel}/model.pk', 'wb') as f:
            pickle.dump(model, f)
        with open(f'users/{user_id}/{channel}/tfidf.pk', 'wb') as f:
            pickle.dump(tfidf, f)


def load_model(user_id: [int, str], channel: str):
    """Загружает модель и tfidf"""
    with locks[user_id]:
        with open(f'users/{user_id}/{channel}/model.pk', 'rb') as f:
            model = pickle.load(f)
        with open(f'users/{user_id}/{channel}/tfidf.pk', 'rb') as f:
            tfidf = pickle.load(f)

    return model, tfidf


def load_dataset(user_id: [int, str], channel: str) -> pd.DataFrame:
    """Загружает датасет для дообучения"""
    with locks[user_id]:
        return pd.read_feather(f'users/{user_id}/{channel}/dataset.feather')


def save_dataset(user_id: [int, str], channel: str, dataset: pd.DataFrame) -> None:
    """Сохраняет датасет"""
    with locks[user_id]:
        dataset.to_feather(f'users/{user_id}/{channel}/dataset.feather', index=None)


def load_config(user_id: [int, str], channel: str) -> dict:
    """Загружает файл настроек модели"""
    with locks[user_id]:
        with open(f'users/{user_id}/{channel}/config.json', 'w') as f:
            return json.load(f)


def save_config(user_id: [int, str], channel: str, config: dict):
    """Сохраняет файл настроек модели"""
    with locks[user_id]:
        with open(f'users/{user_id}/{channel}/config.json', 'w') as f:
            json.dump(config, f)
