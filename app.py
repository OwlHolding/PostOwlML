from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import numpy as np
from datetime import datetime
import pandas as pd

from core import files
from core.requests import *
from core.telegram import get_posts
from core.utils import valid_channel, valid_user
from core import ml

app = FastAPI()


@app.post('/register/{user_id}/')
async def register(user_id: int) -> Response:
    """Регистрация новых пользователей"""

    status_code = 201 + (files.register_user(user_id) * 7)

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

    df = files.load_dataset(user_id, channel)
    files.save_dataset(user_id, channel, pd.concat([df, pd.DataFrame({'posts': posts,
                                                                      'labels': [np.nan for _ in range(len(posts))],
                                                                      'confidence': [np.nan for _ in range(len(posts))],
                                                                      'timestamp': [datetime.now() for _ in
                                                                                    range(len(posts))]
                                                                      })]))

    if status_code == 400:
        return Response('Channel Not Exists', status_code=status_code)

    status_code = 201 + (files.register_channel(user_id, channel) * 7)

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
        dataset[dataset['posts'] == request.posts[i]]['labels'] = request.labels[i]

    if request.finetune:

        if (len(dataset) - 10) % 7 == 0:
            ml.finetune(config=config,
                        user_id=user_id,
                        channel=channel,
                        texts_tf_idf=dataset['posts'].tolist(),
                        labels=dataset[dataset['labels'].notna()]['labels'].tolist(),
                        texts=dataset[dataset['posts'].notna()]['posts'].tolist()
                        )
            if
    else:
        if len(request.posts) == 1 or len(request.posts) == 0:
            return Response('Length Required', status_code=411)

        # thread = Thread(target=ml.fit, kwargs={
        #     "texts": request.posts
        #     "labels": request.labels,
        #     "user_id": user_id,
        #     "channel": channel,
        # })
        #
        # thread.start()

        await ml.fit(
            config=config,
            texts=request.posts,
            labels=request.labels,
            user_id=user_id,
            channel=channel,
            posts_tf_idf=dataset['posts'].tolist()
        )

    dataset.confidence = ml.get_confidence(config, dataset.posts, user_id, channel)

    return Response(status_code=202)


@app.post('/predict/{user_id}/{channel}/')
async def predict(user_id: int, channel: str, request: PredictRequest) -> Response:
    """Выбор лучших постов"""

    if not valid_channel(user_id, channel):
        return Response('User Not Found', status_code=404)

    posts, status_code = await get_posts(channel, 50, request.time)

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
        "posts": response,
        "markup": dataset[dataset.labels.isna()].sort_values(by="confidence")[0],
    }

    if config['drop']:
        dataset = dataset.sort_values(by="timestamp").drop(index=0)
        print(dataset['timestamp'])
    elif len(dataset) + 1 >= 1000:
        config['drop'] = True

    files.save_dataset(user_id, channel, pd.concat([dataset, pd.DataFrame({'posts': posts[-1],
                                                                           'labels': np.nan,
                                                                           'confidence': np.nan,
                                                                           'timestamp': datetime.now()
                                                                           })]))

    return JSONResponse(
        content=content,
        status_code=200
    )
