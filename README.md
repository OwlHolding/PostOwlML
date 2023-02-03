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

### POST <https://$url/train/$id>

Создать модель и обучающую выборку для запрошенного канала.

### Запрос

```json
{
    "channel": "forbesrussia"
}
```

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
Bad Request 400
```

## Тренировка модели

### PUT <https://$url/train/$id>

Запустить тренировку модели.

### Запрос

```json
{
    "posts": [
        "Доллар взлетел до небес",
        "Евро упал до земли"
    ],

    "labels": [1, 0],
  
    "channel": "forbesrussia"
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
Bad Request 400
```

## Выбор лучших постов

### POST <https://$url/predict/$id>

Выбрать *count* лучших постов с канала до *time* (минут) вчерашнего дня и прислать в порядке убывания полезности.

### Запрос

```json
{
    "channel": "forbesrussia",
    "time": 720,
    "count": 2
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
Channel Not Found 404
```

```
Bad Request 400
```
