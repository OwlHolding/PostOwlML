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


@app.post('/train/{user_id}/')
async def create_model(user_id: int, request: ChannelRequest) -> Response:
    """Создание модели и обучающей выборки для запрошенного канала."""

    if not valid_user(user_id):
        return Response('User Not Found', status_code=404)

    content = {
        'posts': get_posts(request.channel, 10, None)
    }

    return JSONResponse(
        content=content,
        status_code=files.register_channel(user_id, request.channel)
    )


@app.put('/train/{user_id}/')
async def train(user_id: int, request: TrainRequest) -> Response:
    """Обучение модели"""

    if not valid_channel(user_id, request.channel):
        return Response('User Not Found', status_code=404)

    dataset = {
        "text": request.posts,
        "label": request.labels
    }

    ml.fit(
        dataset,
        all_texts=request.posts,
        path_model=f"users/{user_id}/{request.channel}/model.pk",
        path_tfidf=f"users/{user_id}/{request.channel}/tfidf.pk"
    )

    return Response(status_code=202)


@app.post('/predict/{user_id}/')
async def predict(user_id: int, request: PredictRequest) -> Response:
    """Выбор лучших постов"""

    if not valid_channel(user_id, request.channel):
        return Response('User Not Found', status_code=404)

    posts = get_posts(request.channel, 10, request.time)

    output = ml.predict(
        posts,
        path_model=f"users/{user_id}/{request.channel}/model.pk",
        path_tfidf=f"users/{user_id}/{request.channel}/tfidf.pk"
    )
    content = {
        "utility": output.tolist()
    }

    return JSONResponse(
        content=content,
        status_code=200
    )
