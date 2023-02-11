from pydantic import BaseModel


class TrainRequest(BaseModel):

    posts: list[str]
    labels: list[int]


class PredictRequest(BaseModel):

    time: int
