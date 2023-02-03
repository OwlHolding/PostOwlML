"""Модуль управления файлами для пар пользователь:канал"""
import os
import multiprocessing

lock = multiprocessing.RLock()


def register_channel(user_id: [int, str], channel: str) -> int:
    """Добавляет канал существующему пользователю"""
    if not os.path.exists(f'users/{user_id}/{channel}'):
        os.makedirs(f'users/{user_id}/{channel}')
        with open(f'users/{user_id}/{channel}/model.pk', 'w') as f:
            pass
        with open(f'users/{user_id}/{channel}/tfidf.pk', 'w') as f:
            pass
        return 200

    return 208


def register_user(user_id: [int, str]) -> int:
    """Регистрирует нового пользователя в системе"""
    if not os.path.exists(f'users/{user_id}'):
        os.makedirs(f'users/{user_id}')
        return 200

    return 208
