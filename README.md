# Архитектура запросов

## Регистрация пользователя

### POST <https://$url/register/$id>

Зарегистрировать нового пользователя.

### Ответ успешно

```
Created 201
Already Reported 208
```

### Ответ ошибка

```
Internal Server Error 500
```

## Регистрация нового канала

### POST <https://$url/regchannel/$id/$channel/>

Создать модель и обучающую выборку для запрошенного канала.

### Ответ успешно

```
Created 201
Already Reported 208
```

```json
{
    "posts": [
        "Доллар взлетел до небес",
        "Евро упало до земли"
    ]
}
```

### Ответ ошибка

```
Internal Server Error 500
```

```
User Not Found 404
```

```
Channel Not Exists 400
```

```
Bad Request 422
```

## Тренировка модели

### POST <https://$url/train/$id/$channel/>

Запустить тренировку модели.

### Запрос

```json
{
    "posts": [
        "Доллар взлетел до небес",
        "Евро упал до земли"
    ],

    "labels": [1, 0],
    
    "finetune": false
}
```

### Ответ успешно

```
Accepted 202 
```

### Ответ ошибка

```
Internal Server Error 500
```

```
User Not Found 404
```

```
Bad Request 422
```

```
Length Required 411
```

## Выбор лучших постов

### POST <https://$url/predict/$id/$channel/>

Выбрать *count* лучших постов с канала до *time* (минут) вчерашнего дня и прислать в порядке убывания полезности.

### Запрос

```json
{
    "time": 720,
}
```

### Ответ успешно

```
OK 200
```

```json
{
    "posts": [
        "Доллар взлетел до небес",
        "Евро упал до земли"
    ],
    
    "markup": "Как нарисовать сову"
}
```

### Ответ ошибка

```
Internal Server Error 500
```

```
User Not Found 404
```

```
Channel Not Found 400
```

```
Bad Request 422
```
