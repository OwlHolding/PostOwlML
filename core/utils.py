import os


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
