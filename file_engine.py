"""Модуль управления файлами для пар пользователь:канал"""
import os
import shutil
import multiprocessing
import torch
from torch.nn.functional import cosine_similarity

lock = multiprocessing.RLock()


def register(user, channel):
    """Регистрирует новую пару пользователь:канал"""
    if not os.path.exists(f'users/{user}/{channel}'):
        os.makedirs(f'users/{user}/{channel}')
    shutil.copyfile('static/negative.pt', f'users/{user}/{channel}/negative.pt')
    shutil.copyfile('static/positive.pt', f'users/{user}/{channel}/positive.pt')


def predict(user, channel, p_text):
    """Возвращает полезность предобработанного текста для пары пользователь:канал"""
    with lock:
        if not os.path.exists(f'users/{user}/{channel}/'):
            register(user, channel)

        positive = torch.load(f'users/{user}/{channel}/positive.pt')

    result = []
    for text in p_text:
        # взятие косинусной близости и перевод в интервал [0, 1]
        result.append((cosine_similarity(p_text, positive).item() + 1) / 2)
    return result


def async_fit(user, channel, data, labels):
    """Асинхронно обучает модель"""
    multiprocessing.Process(target=fit,
                            args=(user, channel, data, labels, lock)).start()


def fit(user, channel, data, labels, lock):
    """Обучает и сохраняет модель"""
    with lock:
        if not os.path.exists(f'users/{user}/{channel}.npy'):
            register(user, channel)
        positive = torch.load(f'users/{user}/{channel}/positive.pt')
        negative = torch.load(f'users/{user}/{channel}/negative.pt')

    positive = torch.rand([768])  # функции заглушки
    negative = torch.rand([768])

    with lock:
        torch.save(positive, f'users/{user}/{channel}/positive.pt')
        torch.save(negative, f'users/{user}/{channel}/negative.pt')
