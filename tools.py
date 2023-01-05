'''Вспомогательные инструменты'''
import io
import numpy as np

def validate_reg_request(request):
    "Проверяет правильность запроса регистрации нового пользователя"
    if 'token' not in request or 'channel' not in request or 'user' not in request:
        return False
    if request['token'] and request['channel'] and request['user']:
        return True
    return False

def validate_pred_request(request):
    "Проверяет правильность запроса инференса"
    if 'text' not in request or not request['text'] or not\
            isinstance(request['text'], list):
        return False
    return validate_reg_request(request)

def validate_fit_request(request):
    res = True
    if 'labels' not in request or not request['labels'] or not\
            isinstance(request['labels'], list):
        res = False
    res = res and validate_pred_request(request)
    if res and len(request['labels']) != len(request['text']):
        res = False
    return res

def serialize_numpy(array):
    '''Сжимает numpy массив и преобразует его в строку'''
    stream = io.BytesIO()
    np.save(stream, array)
    return stream.getvalue()

def deserialize_numpy(np_str):
    '''Выполняет десериализацию numpy массива из строки'''
    stream = io.BytesIO()
    stream.write(np_str)
    stream.seek(0)
    return np.load(stream)