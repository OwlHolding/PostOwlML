"""Модуль для поддержки Bert"""
import copy
import threading
import time  # Выключить после тестов
import functools
import transformers
import numpy as np  # Выключить после тестов

cache_size = 100  # Изменить в соответствии с объемом ОЗУ

lock = threading.RLock()

tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")


@functools.lru_cache(maxsize=cache_size)
def extract(text):
    """Извлекает признаки из текста """
    with lock:
        # inputs = tokenizer(text, return_tensors="pt")
        # return model(**inputs).last_hidden_state.detach().numpy()
        time.sleep(3)  # Имитация работы BERTа
        return np.array([int.from_bytes(text.encode(), 'big')])  # Для тестов


def turbo_extract(text):  # Не включать!!! Крайне опасно
    """Экспериментальная функция!!! Параллельное извлечение признаков на CPU"""
    turbo_tokenizer = copy.deepcopy(tokenizer)
    turbo_model = copy.deepcopy(model)
    inputs = turbo_tokenizer(text, return_tensors="pt")
    return turbo_model(**inputs).last_hidden_state.detach().numpy()
