from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse

import logging
import json

import ml
from schemes import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(asctime)s - %(message)s", filename="log.txt",
                    filemode="w")

app = FastAPI(docs_url=None)

with open("config.json", 'rb') as file:
    config = json.load(file)

WHITELIST = config['WHITELIST']


def firewall(request):
    if len(WHITELIST) == 0:
        return True
    ip = request.client.host
    print(ip)
    if ip is None:
        return True
    if ip in WHITELIST:
        return True

    return False


@app.post("/add-user/{user_id}/")
async def add_user(user_id: int, request: Request) -> Response:
    if not firewall(request):
        return Response(status_code=403)

    status_code = (not ml.add_user(user_id)) * 7 + 201

    logging.info(f"{user_id} add user - {status_code}")

    return Response(status_code=status_code)


@app.post("/del-user/{user_id}/")
async def del_user(user_id: int, request: Request) -> Response:
    if not firewall(request):
        return Response(status_code=403)

    status_code = (not ml.del_user(user_id)) * 3 + 205

    logging.info(f"{user_id} del user - {status_code}")

    return Response(status_code=status_code)


@app.post("/add-channel/{user_id}/{channel}/")
async def add_channel(user_id: int, channel: str, request: Request) -> Response:
    if not firewall(request):
        return Response(status_code=403)

    status_code = (not ml.add_channel(user_id, channel)) * 7 + 201

    logging.info(f"{user_id} add channel {channel} - {status_code}")

    return Response(status_code=status_code)


@app.post("/del-channel/{user_id}/{channel}/")
async def del_channel(user_id: int, channel: str, request: Request) -> Response:
    if not firewall(request):
        return Response(status_code=403)

    status_code = (not ml.del_channel(user_id, channel)) * 3 + 205

    logging.info(f"{user_id} del channel {channel} - {status_code}")

    return Response(status_code=status_code)


@app.post("/predict/")
async def predict(request: PredictRequest, row_request: Request) -> Response:
    if not firewall(row_request):
        return Response(status_code=403)

    users = ml.predict(request.post, request.channel, request.users)

    if len(users) == 0:
        logging.info(f"predict request for channel {request.channel} - 400")
        return Response(status_code=400)

    logging.info(f"predict request for channel {request.channel} - 202")
    return JSONResponse(
        content={
            "users": users
        },
        status_code=202
    )


@app.post("/train/{user_id}/{channel}/")
async def train(user_id: int, channel: str, request: TrainRequest, row_request: Request):
    if not firewall(row_request):
        return Response(status_code=403)

    logging.info(f"{user_id} send label for channel {channel} - 202")

    ml.train(user_id, channel, request.text, request.label)

    logging.info(f"{user_id} label for channel {channel} saved")

    return Response(status_code=202)
