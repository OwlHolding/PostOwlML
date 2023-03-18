import os
from bs4 import BeautifulSoup
import time

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


def retry(times, exceptions, time_sleep):
    """Декоратор для повторения функции"""

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    time.sleep(time_sleep)
                    attempt += 1
            return func(*args, **kwargs)
        return newfn
    return decorator


def remove_tags(text: str) -> str:
    """Удаление нечитаемых тегов"""

    html = BeautifulSoup(text, 'html.parser')

    container = html.find('div')
    keep = []

    for node in container.descendants:
        if not node.name or node.name in tag_list:
            keep.append(node)
    return "".join(map(str, keep))

