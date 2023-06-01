from pydantic import BaseModel


class PredictRequest(BaseModel):
    channels: list
    count: int
    time: int


class TrainRequest(BaseModel):
    text: str
    label: bool
