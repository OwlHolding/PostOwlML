from pydantic import BaseModel


class ChannelRequest(BaseModel):

    channel: str


class TrainRequest(BaseModel):

    posts: list[str]
    labels: list[int]
    channel: str


class PredictRequest(BaseModel):

    channel: str
    time: int
    count: int
