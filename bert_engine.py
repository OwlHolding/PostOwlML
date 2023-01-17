"""Модуль для поддержки Bert"""
import copy
import threading
import functools
from transformers import AutoTokenizer, AutoModel, BertModel
import settings
from torch.nn.functional import normalize

lock = threading.RLock()

model = AutoModel.from_pretrained(settings.MODEL)
tokenizer = AutoTokenizer.from_pretrained(settings.MODEL)


@functools.lru_cache(maxsize=settings.CASH_SIZE)
def extract(text):
    """Извлекает признаки из текста """
    with lock:
        return normalize(model.forward(**tokenizer([text], return_tensors="pt"))['last_hidden_state'][:, 0, :], dim=0)


def turbo_extract(text):
    """Экспериментальная функция!!! Параллельное извлечение признаков на CPU"""
    turbo_tokenizer = copy.deepcopy(tokenizer)
    turbo_model = copy.deepcopy(model)
    inputs = turbo_tokenizer(text, return_tensors="pt")
    return turbo_model(**inputs).last_hidden_state.detach().numpy()
