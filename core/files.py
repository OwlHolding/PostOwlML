"""Модуль управления файлами для пар пользователь:канал"""
import os
import multiprocessing
import pickle

lock = multiprocessing.RLock()


def register_channel(user_id: [int, str], channel: str) -> bool:
    """Добавляет канал существующему пользователю"""
    if not os.path.exists(f'users/{user_id}/{channel}'):
        os.makedirs(f'users/{user_id}/{channel}')
        with open(f'users/{user_id}/{channel}/model.pk', 'w') as f:
            pass
        with open(f'users/{user_id}/{channel}/tfidf.pk', 'w') as f:
            pass
        return False

    return True


def register_user(user_id: [int, str]) -> bool:
    """Регистрирует нового пользователя в системе"""
    if not os.path.exists(f'users/{user_id}'):
        os.makedirs(f'users/{user_id}')
        return False

    return True


def save_model(path: str, model, tfidf):
    """Сохраняет модель и tfidf"""
    with lock:
        with open(path + "/model.pk", 'wb') as f:
            pickle.dump(model, f)
        with open(path + "/tfidf.pk", 'wb') as f:
            pickle.dump(tfidf, f)


def load_model(path: str):
    """Загружает модель и tfidf"""
    with lock:
        with open(path + "/model.pk", 'rb') as f:
            model = pickle.load(f)
        with open(path + "/tfidf.pk", 'rb') as f:
            tfidf = pickle.load(f)

    return model, tfidf
