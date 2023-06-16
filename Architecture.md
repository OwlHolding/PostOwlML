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

## POST User

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

## POST Channel

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

### POST <http://$url/predict/>

Запрос:
```json
{
  "post": "post text",
  "channel": "post channel",
  "users": [1, 2, 3, 4]
}
```

Ответ:
```json
{
  "users": [1, 3]
}
```


### Успешно

```
202 Accepted 
400 No users
```

### Ошибка

```
500 Internal Server Error 
422 Unprocessable Entity
```

## Train

### POST <http://$url/predict/$user-id/$channel/>

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

