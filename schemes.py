from pydantic import BaseModel


class PredictRequest(BaseModel):
    post: str
    channel: str
    users: list[int]


class TrainRequest(BaseModel):
    text: str
    label: bool
