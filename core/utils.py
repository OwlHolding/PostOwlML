import os
import time
import numpy as np
import re

tag_list = [
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "span", "tg-spoiler", "code", "pre", "a", "img"
]


def valid_user(user_id: int) -> bool:
    """Проверка существования пользователя"""
    if os.path.exists(f'users/{user_id}/'):
        return True

    return False


def valid_channel(user_id: int, channel: str) -> bool:
    """Проверка существования пары пользователь канал"""

    if os.path.exists(f'users/{user_id}/{channel}/'):
        return True

    return False


def retry(times: int, exceptions, min_delay: int, max_delay: int, factor=2, scale=1):
    """Декоратор для повторения функции"""

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            delay = min_delay
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    time.sleep(delay)
                    delay = min(delay * factor, max_delay)
                    delay = np.random.normal(delay, scale=scale)
                    attempt += 1
            return func(*args, **kwargs)

        return newfn

    return decorator


def remove_tags(text: str) -> str:
    """Удаление нечитаемых тегов"""
    return re.sub(fr"<\/?(?!{'|'.join(tag_list)})[^>]*>", "", text)
