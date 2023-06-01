# Архитектура взаимодействия с главным сервером

## Add User

### POST <http://$url/add-user/$user-id/>

### Успешно 

```
201 Created 
208 Already Reported
```

### Ошибка

```
500 Internal Server Error 
```

## Del User

### DELETE <http://$url/del-user/$user-id/>

### Успешно 

```
205 Reset Content
208 Already Reported
```
### Ошибка

```
500 Internal Server Error 
404 User Not Found 
```

## Add Channel

### POST <http://$url/add-channel/$user-id/$channel/>

### Успешно 

```
201 Created 
208 Already Reported
```

### Ошибка

```
500 Internal Server Error 
```

## Del Channel

### DELETE <http://$url/add-channel/$user-id/$channel/>

### Успешно 

```
205 Reset Content
208 Already Reported
```
### Ошибка

```
500 Internal Server Error 
404 User Not Found 
400 Channel Not Found
```

## Predict

### GET <http://$url/predict/$user-id/>

Запрос:
```json
{
  "channels": ["test-channel"],
  "count": 10,
  "time": 1
}
```

Ответ:
```json
{
  "feed": ["first-text", "second-text"]
}
```


### Успешно

```
202 Accepted 
```

### Ошибка

```
500 Internal Server Error 
404 User Not Found
400 Channel Not Found
422 Unprocessable Entity
```

## Train

### PUT <http://$url/predict/$user-id/$channel/>

Запрос:
```json
{
  "text":"post-text",
  "label": 1
}
```

### Успешно

```
202 Accepted 
```

### Ошибка

```
500 Internal Server Error 
404 User Not Found
400 Channel Not Found
422 Unprocessable Entity
```

