from sklearnex import patch_sklearn
patch_sklearn()

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import numpy as np
from datetime import datetime
import pandas as pd
import logging
from threading import Thread

from core import files
from core.request import *
from core.telegram import get_posts
from core.utils import valid_channel, valid_user
from core import ml

app = FastAPI()

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:     %(asctime)s - %(message)s", filename="log.txt",
                    filemode="w")


def save_confidence(config, dataset, user_id, channel) -> None:
    logging.debug(dataset)
    logging.debug(type(config))
    logging.debug(config)

    dataset.confidence = ml.get_confidence(config, dataset.posts, user_id, channel)
    files.save_dataset(user_id, channel, dataset)
    logging.debug(f"Dataset for user {user_id}:{channel} saved")


@app.post('/register/{user_id}/')
async def register(user_id: int) -> Response:
    """Регистрация новых пользователей"""

    status_code = 201 + (files.register_user(user_id) * 7)

    if status_code == 201:
        logging.info(f"Created new user {user_id}")

    return Response(
        content="",
        status_code=status_code
    )


@app.post('/regchannel/{user_id}/{channel}/')
async def create_model(user_id: int, channel: str) -> Response:
    """Создание модели и обучающей выборки для запрошенного канала."""

    if not valid_user(user_id):
        return Response('User Not Found', status_code=404)

    posts, status_code = await get_posts(channel, 50, 0)
    logging.debug(f"Successfully received {len(posts)} posts from the channel {channel}")

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
    logging.debug(f"The dataset for {user_id}:{channel} has been updated")

    content = {
        'posts': posts[:10]
    }

    return JSONResponse(
        content=content,
        status_code=status_code
    )


@app.post('/train/{user_id}/{channel}/')
async def train(user_id: int, channel: str, request: TrainRequest) -> Response:
    """Обучение модели"""

    if not valid_channel(user_id, channel):
        return Response('User Not Found', status_code=404)

    dataset = files.load_dataset(user_id, channel)
    config = files.load_config(user_id, channel)

    for i in range(len(request.posts)):
        dataset.loc[dataset['posts'] == request.posts[i], 'labels'] = request.labels[i]

    logging.info(f'Dataset Size for user {user_id}:{channel} is {len(dataset)}')

    if request.finetune:

        if (len(dataset[dataset['labels'].notna()]) - 10) % 7 == 0:
            logging.info(f"Started Owl Learning step for user {user_id}:{channel}")
            ml.finetune(config=config,
                        user_id=user_id,
                        channel=channel,
                        texts_tf_idf=dataset['posts'].tolist(),
                        labels=dataset[dataset['labels'].notna()]['labels'].tolist(),
                        texts=dataset.loc[dataset['labels'].notna(), 'posts'].tolist()
                        )

    else:

        if len(request.posts) == 1 or len(request.posts) == 0:
            return Response('Length Required', status_code=411)

        logging.info(f"Started training model for user {user_id}:{channel}")

        # tr = Thread(target=ml.fit, args={"config": config,
        #                                  "texts": request.posts,
        #                                  "labels": request.labels,
        #                                  "user_id": user_id,
        #                                  "channel": channel,
        #                                  "texts_tf_idf": dataset['posts'].tolist()})
        # tr.start()

        ml.fit(config=config,
               texts=request.posts,
               labels=request.labels,
               user_id=user_id,
               channel=channel,
               texts_tf_idf=dataset['posts'].tolist())

        logging.info(f"Model trained for user {user_id}:{channel}")

    logging.info(f"Started get_confidence {user_id}:{channel}")

    tr = Thread(target=save_confidence, args=[config, dataset, user_id, channel])
    tr.start()

    return Response(status_code=202)


@app.post('/predict/{user_id}/{channel}/')
async def predict(user_id: int, channel: str, request: PredictRequest) -> Response:
    """Выбор лучших постов"""

    if not valid_channel(user_id, channel):
        return Response('User Not Found', status_code=404)

    posts, status_code = await get_posts(channel, 50, request.time)
    logging.debug(f"Successfully received {len(posts)} posts from the channel {channel}")

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

    output = ml.predict(
        config=config,
        texts=posts,
        user_id=user_id,
        channel=channel,
    )

    response = []
    for i in range(len(posts)):
        if output[i] == 1:
            response.append(posts[i])

    content = {
        "posts": response[:5],
        "markup": dataset[dataset.labels.isna()].sort_values(by="confidence").iloc[0].posts,
    }

    if config['drop']:
        dataset = dataset.sort_values(by="timestamp").drop(index=0)
        logging.debug(f"Row removed from dataset for user {user_id}:{channel}")

    elif len(dataset) + 1 >= 1000:
        config['drop'] = True
        logging.debug(f"Set 'drop' in config for user {user_id}:{channel}")

    files.save_dataset(user_id, channel, pd.concat([dataset, pd.DataFrame({'posts': [posts[-1]],
                                                                           'labels': [np.nan],
                                                                           'confidence': [np.nan],
                                                                           'timestamp': [datetime.now()]
                                                                           })], ignore_index=True))

    return JSONResponse(
        content=content,
        status_code=200
    )
