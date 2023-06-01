from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

import logging

import ml
import telegram
from schemes import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(asctime)s - %(message)s", filename="log.txt",
                    filemode="w")

app = FastAPI()


@app.post("/add-user/{user_id}/")
async def add_user(user_id: int) -> Response:
    status_code = (not ml.add_user(user_id)) * 7 + 201

    logging.info(f"{user_id} add user - {status_code}")

    return Response(status_code=status_code)


@app.delete("/del-user/{user_id}/")
async def del_user(user_id: int) -> Response:
    status_code = (not ml.del_user(user_id)) * 3 + 205

    logging.info(f"{user_id} del user - {status_code}")

    return Response(status_code=status_code)


@app.post("/add-channel/{user_id}/{channel}/")
async def add_channel(user_id: int, channel: str) -> Response:
    status_code = (not ml.add_channel(user_id, channel)) * 7 + 201

    logging.info(f"{user_id} add channel {channel} - {status_code}")

    return Response(status_code=status_code)


@app.delete("/del-channel/{user_id}/{channel}/")
async def del_channel(user_id: int, channel: str) -> Response:
    status_code = (not ml.del_channel(user_id, channel)) * 3 + 205

    logging.info(f"{user_id} del channel {channel} - {status_code}")

    return Response(status_code=status_code)


@app.post("/predict/{user_id}/")
async def predict(user_id: int, request: PredictRequest) -> Response:
    post = telegram.get_posts(request.channels, request.count, request.time)

    feed = ml.predict(user_id, ml.PredictedData(post)).get_feed()

    logging.info(f"{user_id} the feed was created successfully - 202")
    logging.debug(f"Feed Len: {len(feed)}")

    return JSONResponse(
        content={
            "feed": [telegram.remove_tags(text) for text in feed]
        },
        status_code=202
    )


@app.put("/train/{user_id}/{channel}/")
async def train(user_id: int, channel: str, request: TrainRequest):
    logging.info(f"{user_id} send label for channel {channel} - 202")

    ml.train(user_id, channel, request.text, request.label)

    logging.info(f"{user_id} label for channel {channel} saved")

    return Response(status_code=202)
