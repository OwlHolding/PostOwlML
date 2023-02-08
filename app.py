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

    return Response(
        content="",
        status_code=files.register_user(user_id)
    )


@app.post('/regchannel/{user_id}/')
async def create_model(user_id: int, request: ChannelRequest) -> Response:
    """Создание модели и обучающей выборки для запрошенного канала."""

    if not valid_user(user_id):
        return Response('User Not Found', status_code=404)

    content = {
        'posts': await get_posts(request.channel, 10, None)
    }

    return JSONResponse(
        content=content,
        status_code=files.register_channel(user_id, request.channel)
    )


@app.post('/train/{user_id}/')
async def train(user_id: int, request: TrainRequest) -> Response:
    """Обучение модели"""

    if not valid_channel(user_id, request.channel):
        return Response('User Not Found', status_code=404)

    ml.fit(
        texts=request.posts,
        labels=request.labels,
        path=f"users/{user_id}/{request.channel}"
    )

    return Response(status_code=202)


@app.post('/predict/{user_id}/')
async def predict(user_id: int, request: PredictRequest) -> Response:
    """Выбор лучших постов"""

    if not valid_channel(user_id, request.channel):
        return Response('User Not Found', status_code=404)

    posts = await get_posts(request.channel, request.count, request.time)

    output = ml.predict(
        posts,
        path=f"users/{user_id}/{request.channel}",
    )

    content = {
        "posts": sorted(posts, key=lambda x: output[posts.index(x)][1])

    }
    return JSONResponse(
        content=content,
        status_code=200
    )
