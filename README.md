# Архитектура запросов

## Регистрация пользователя

### POST <https://$url/register/$id>

Зарегистрировать нового пользователя.

### Ответ успешно

```json
Created 201
Already Reported 208
```

### Ответ ошибка

```json
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

```json
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

```json
Internal Server Error 500
```

```json
User Not Found 404
```

```json
Channel Not Exists 400
```

```json
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

```json
Accepted 202 
```

### Ответ ошибка

```json
Internal Server Error 500
```

```json
User Not Found 404
```

```json
Bad Request 400
```

## Выбор лучших постов

### GET <https://$url/predict/$id>

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

```json
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

```json
Internal Server Error 500
```

```json
User Not Found 404
```

```json
Channel Not Found 404
```

```json
Bad Request 400
```
