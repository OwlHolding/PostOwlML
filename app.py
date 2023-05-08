from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import numpy as np
from datetime import datetime
import pandas as pd
import logging
from threading import Thread

from core.request import *
from core.telegram import get_posts
from core.utils import remove_tags

import ml
import files

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(asctime)s - %(message)s", filename="log.txt",
                    filemode="w")


def save_confidence(config, dataset, user_id, channel) -> None:
    model, tfidf = files.load_model(user_id, channel, config)

    dataset.confidence = ml.get_confidence(texts=dataset.posts, model=model, tfidf=tfidf)
    files.save_dataset(user_id, channel, dataset)
    logging.info(f"Dataset for user {user_id}:{channel} saved")


@app.post('/register/{user_id}/')
async def register(user_id: int) -> Response:
    """Регистрация новых пользователей"""

    status_code = 201 + (files.register_user(user_id) * 7)

    if status_code == 201:
        logging.info(f"Created new user {user_id}")
    else:
        logging.info(f"Attempt to register a user {user_id}")

    return Response(
        content="",
        status_code=status_code
    )


@app.post('/regchannel/{user_id}/{channel}/')
async def create_model(user_id: int, channel: str) -> Response:
    """Создание модели и обучающей выборки для запрошенного канала."""

    if not files.valid_user(user_id):
        return Response('User Not Found', status_code=404)
    try:
        posts, status_code = get_posts(channel, 50, 0)

    except Exception as e:
        logging.error(e)
        return Response('Unsupported channel', status_code=400)

    logging.info(f"Successfully received {len(posts)} posts from the channel {channel}")

    if status_code == 400:
        return Response('Channel Not Exists', status_code=status_code)

    status_code = 201 + (files.register_channel(user_id, channel) * 7)

    if status_code == 201:
        logging.info(f"Successfully registered a new channel for {user_id}:{channel}")

    df = files.load_dataset(user_id, channel)
    files.save_dataset(user_id, channel, pd.concat([df, pd.DataFrame({'posts': posts,
                                                                      'labels': [np.nan for _ in range(len(posts))],
                                                                      'confidence': [np.nan for _ in range(len(posts))],
                                                                      'timestamp': [datetime.now() for _ in
                                                                                    range(len(posts))]
                                                                      })]))
    logging.info(f"The dataset for {user_id}:{channel} has been updated")

    content = {
        'posts': [remove_tags(post) for post in posts[:10]]
    }

    return JSONResponse(
        content=content,
        status_code=status_code
    )


@app.post('/train/{user_id}/{channel}/')
async def train(user_id: int, channel: str, request: TrainRequest) -> Response:
    """Обучение модели"""

    if not files.valid_channel(user_id, channel):
        return Response('User Not Found', status_code=404)

    dataset = files.load_dataset(user_id, channel)
    config = files.load_config(user_id, channel)

    for i in range(len(request.posts)):
        dataset.loc[dataset['posts'] == request.posts[i], 'labels'] = request.labels[i]

    logging.info(f'Dataset Size for user {user_id}:{channel} is {len(dataset)}')

    if request.finetune:

        if (dataset['labels'].notna().sum() - 10) % 6 == 0:
            logging.info(f"Started Owl Learning step for user {user_id}:{channel}")

            model, tfidf = ml.finetune(config=config,
                                       user_id=user_id,
                                       channel=channel,
                                       texts_tf_idf=dataset['posts'].tolist(),
                                       labels=dataset[dataset['labels'].notna()]['labels'].tolist(),
                                       texts=dataset.loc[dataset['labels'].notna(), 'posts'].tolist()
                                       )

            files.save_model(user_id, channel, model, tfidf, config)

    else:

        if len(request.posts) == 1 or len(request.posts) == 0:
            return Response('Length Required', status_code=411)

        logging.info(f"Started training model for user {user_id}:{channel}")

        model, tfidf = ml.fit(config=config,
                              texts=request.posts,
                              labels=request.labels,
                              texts_tf_idf=dataset['posts'].tolist())

        files.save_model(user_id, channel, model, tfidf, config)

        logging.info(f"Model trained for user {user_id}:{channel}")

    logging.info(f"Started get_confidence {user_id}:{channel}")

    tr = Thread(target=save_confidence, args=[config, dataset, user_id, channel])
    tr.start()

    return Response(status_code=202)


@app.post('/predict/{user_id}/{channel}/')
async def predict(user_id: int, channel: str, request: PredictRequest) -> Response:
    """Выбор лучших постов"""

    if not files.valid_channel(user_id, channel):
        return Response('User Not Found', status_code=404)

    posts, status_code = get_posts(channel, 50, request.time)
    logging.info(f"Successfully received {len(posts)} posts from the channel {channel}")

    dataset = files.load_dataset(user_id, channel)
    config = files.load_config(user_id, channel)

    if status_code == 400:
        return Response('Channel Not Exists', status_code=status_code)

    if len(posts) == 0:
        return JSONResponse(
            content={
                'posts': [],
                'markup': ""
            },
            status_code=200
        )

    model, tfidf = files.load_model(user_id, channel, config)

    output = ml.predict(
        texts=posts,
        model=model,
        tfidf=tfidf,
    )

    response = []
    for i in range(len(posts)):
        if output[i] == 1:
            response.append(posts[i])

    content = {
        "posts": [remove_tags(post) for post in response[:5]],
        "markup": remove_tags(dataset[dataset.labels.isna()].sort_values(by="confidence").iloc[0].posts),
    }

    if config['drop']:
        dataset = dataset.sort_values(by="timestamp").drop(index=0)
        logging.debug(f"Row removed from dataset for user {user_id}:{channel}")

    elif len(dataset) + 1 >= 1000:
        config['drop'] = True
        logging.info(f"Set 'drop' in config for user {user_id}:{channel}")

    files.save_dataset(user_id, channel, pd.concat([dataset, pd.DataFrame({'posts': [i for i in response[:5]],
                                                                           'labels': [np.nan for _ in response[:5]],
                                                                           'confidence': [np.nan for _ in response[:5]],
                                                                           'timestamp': [datetime.now() for _ in response[:5]]
                                                                           })], ignore_index=True))

    return JSONResponse(
        content=content,
        status_code=200
    )
