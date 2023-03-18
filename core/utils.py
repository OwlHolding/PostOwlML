import os

tag_list = [
    "<b", "<strong", "<i", "<em", "<u", "<ins", "<s", "<strike", "<del", "<span", "<tg-spoiler", "<b", "<code", "<pre", "<a"
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


def retry(times, exceptions):
    """Декоратор для повторения функции"""

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    attempt += 1
            return func(*args, **kwargs)
        return newfn
    return decorator


def remove_tags(text):
    """Удаление нечитаемых тегов"""

    stack = []

    answer = ""

    for word in text.split():
        if "<" in word and word not in tag_list:
            stack.append(word)

        if ">" in word:
            stack.pop()

        elif len(stack) == 0:
            answer += f" {word} "

    return answer