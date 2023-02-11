from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

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

    posts, status_code = await get_posts(channel, 10, 0)

    if status_code == 400:
        return Response('Channel Not Exists', status_code=status_code)

    status_code = 201 + (files.register_channel(user_id, channel) * 7)

    content = {
        'posts': posts
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

    if len(request.posts) == 1 or len(request.posts) == 0:
        return Response('Length Required', status_code=411)

    await ml.fit(
        texts=request.posts,
        labels=request.labels,
        path=f"users/{user_id}/{channel}"
    )

    return Response(status_code=202)


@app.post('/predict/{user_id}/{channel}/')
async def predict(user_id: int, channel: str, request: PredictRequest) -> Response:
    """Выбор лучших постов"""

    if not valid_channel(user_id, channel):
        return Response('User Not Found', status_code=404)

    posts, status_code = await get_posts(channel, 50, request.time)

    if status_code == 400:
        return Response('Channel Not Exists', status_code=status_code)

    if len(posts) == 0:
        return JSONResponse(
            content={
                'posts': []
            },
            status_code=200
        )

    output = ml.predict(
        posts,
        path=f"users/{user_id}/{channel}",
    )

    response = []
    for i in range(len(posts)):
        if output[i] == 1:
            response.append(posts[i])

    content = {
        "posts": response
    }

    return JSONResponse(
        content=content,
        status_code=200
    )
