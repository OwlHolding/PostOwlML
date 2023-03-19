from pydantic import BaseModel


class TrainRequest(BaseModel):
    posts: list[str]
    labels: list[int]
    finetune: bool


class PredictRequest(BaseModel):
    time: int
