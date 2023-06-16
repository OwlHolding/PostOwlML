from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

import logging

import ml
from schemes import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(asctime)s - %(message)s", filename="log.txt",
                    filemode="w")

app = FastAPI()


@app.post("/add-user/{user_id}/")
async def add_user(user_id: int) -> Response:
    status_code = (not ml.add_user(user_id)) * 7 + 201

    logging.info(f"{user_id} add user - {status_code}")

    return Response(status_code=status_code)


@app.post("/del-user/{user_id}/")
async def del_user(user_id: int) -> Response:
    status_code = (not ml.del_user(user_id)) * 3 + 205

    logging.info(f"{user_id} del user - {status_code}")

    return Response(status_code=status_code)


@app.post("/add-channel/{user_id}/{channel}/")
async def add_channel(user_id: int, channel: str) -> Response:
    status_code = (not ml.add_channel(user_id, channel)) * 7 + 201

    logging.info(f"{user_id} add channel {channel} - {status_code}")

    return Response(status_code=status_code)


@app.post("/del-channel/{user_id}/{channel}/")
async def del_channel(user_id: int, channel: str) -> Response:
    status_code = (not ml.del_channel(user_id, channel)) * 3 + 205

    logging.info(f"{user_id} del channel {channel} - {status_code}")

    return Response(status_code=status_code)


@app.post("/predict/")
async def predict(request: PredictRequest) -> Response:

    users = ml.predict(request.post, request.channel, request.users)

    logging.info(f"predict request for channel {request.channel} - 202")
    logging.debug(f"{request.users} -> {users}")

    return JSONResponse(
        content={
            "users": users
        },
        status_code=202
    )


@app.post("/train/{user_id}/{channel}/")
async def train(user_id: int, channel: str, request: TrainRequest):
    logging.info(f"{user_id} send label for channel {channel} - 202")

    ml.train(user_id, channel, request.text, request.label)

    logging.info(f"{user_id} label for channel {channel} saved")

    return Response(status_code=202)
