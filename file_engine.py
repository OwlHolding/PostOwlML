"""Модуль управления файлами для пар пользователь:канал"""
import os
import shutil
import multiprocessing
import random  # Для тестов
import numpy as np

lock = multiprocessing.RLock()


def register(user, channel):
    "Регистрирует новую пару пользователь:канал"
    if not os.path.exists(f'users/{user}'):
        os.makedirs(f'users/{user}')
    shutil.copyfile('static/default.npy', f'users/{user}/{channel}.npy')


def predict(user, channel, p_text):
    "Возвращает полезность предобработанного текста для пары пользователь:канал"
    with lock:
        if not os.path.exists(f'users/{user}/{channel}.npy'):
            register(user, channel)
        pair_data = np.load(f'users/{user}/{channel}.npy')
    # Здесь происходят вычисления
    return random.random()  # Убью, если увижу это в продакте :)


def async_fit(user, channel, data, labels):
    "Асинхронно обучает модель"
    multiprocessing.Process(target=fit,
                            args=(user, channel, data, labels, lock)).start()


def fit(user, channel, data, labels, lock):
    "Обучает и сохраняет модель"
    with lock:
        if not os.path.exists(f'users/{user}/{channel}.npy'):
            register(user, channel)
        pair_data = np.load(f'users/{user}/{channel}.npy')
    # Здесь происходит обучение, но кода пока нет
    with lock:
        np.save(f'users/{user}/{channel}.npy', pair_data)
