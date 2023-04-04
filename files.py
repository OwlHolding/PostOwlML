import json
import pickle
import pandas as pd
from core.orm import DataBase


db = DataBase("users", main_node='user_id', child_node='channel')


def valid_user(user_id: [int, str]) -> bool:
    return db.exist_main_node(user_id=user_id)


def valid_channel(user_id: [int, str], channel: str) -> bool:
    return db.exist_child_node(user_id=user_id, channel=channel)


def register_user(user_id: [int, str]) -> bool:
    """Регистрирует нового пользователя в системе"""
    if not db.exist_main_node(user_id=user_id):
        db.set_main_node(user_id=user_id)
        return False

    return True


def register_channel(user_id: [int, str], channel: str) -> bool:
    """Добавляет канал существующему пользователю"""

    if not db.exist_child_node(channel=channel, user_id=user_id):
        db.set_child_node(user_id=user_id, channel=channel)

        db.save_file(user_id=user_id, channel=channel, file_name='model.pk', bfile=b'')
        db.save_file(user_id=user_id, channel=channel, file_name='tfidf.pk', bfile=b'')

        dataset = pd.DataFrame(columns=['posts', 'labels', 'confidence', 'timestamp'])
        config = {
            "model": "SVM",
            "drop": False,
        }

        save_dataset(user_id, channel, dataset)
        save_config(user_id, channel, config)
        return False

    return True


def save_model(user_id: [int, str], channel: str, model, tfidf, config: dict) -> None:
    """Сохраняет модель и tfidf"""

    db.save_file(user_id=user_id, channel=channel, file_name="tfidf.pk", bfile=pickle.dumps(tfidf))
    db.save_file(user_id=user_id, channel=channel, file_name="model.pk", bfile=pickle.dumps(model))


def load_model(user_id: [int, str], channel: str, config: dict):
    """Загружает модель и tfidf"""

    tfidf = pickle.loads(db.load_file(user_id=user_id, channel=channel, file_name="tfidf.pk"))
    model = pickle.loads(db.load_file(user_id=user_id, channel=channel, file_name="model.pk"))

    return model, tfidf


def load_dataset(user_id: [int, str], channel: str) -> pd.DataFrame:
    """Загружает датасет для дообучения"""
    with db.locks[user_id]:
        return pd.read_feather(f'{db.name}/{user_id}/{channel}/dataset.feather')


def save_dataset(user_id: [int, str], channel: str, dataset: pd.DataFrame) -> None:
    """Сохраняет датасет"""
    with db.locks[user_id]:
        dataset = dataset.reset_index(drop=True)
        dataset.to_feather(f'{db.name}/{user_id}/{channel}/dataset.feather')


def load_config(user_id: [int, str], channel: str) -> dict:
    """Загружает файл настроек модели"""
    with db.locks[user_id]:
        with open(f'{db.name}/{user_id}/{channel}/config.json', 'r') as f:
            return json.load(f)


def save_config(user_id: [int, str], channel: str, config: dict):
    """Сохраняет файл настроек модели"""
    with db.locks[user_id]:
        with open(f'{db.name}/{user_id}/{channel}/config.json', 'w') as f:
            json.dump(config, f)
