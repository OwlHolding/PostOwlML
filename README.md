# PostOwlMlServer

Сервер для обработки запросов к ML-модели.

## Install 

Скопируйте репозиторий и перейдите в рабочую папку

Для установки зависимостей выполните:
```shell
pip3 install -r requirements.txt
```

Если хотите выполнить тесты запустите:
```shell
pytest test.py
```

## Using 

Для поднятия сервера выполните:
```shell
uvicorn app:app --host your_host_name --port your_port
```

Все готово, сервер успешно запущен. Для получения подробной информации о структуре и адресе запросов, смотрите файл `Architecture.md`
