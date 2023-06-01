from .structure import PredictedData

def add_user(user_id: int) -> bool:
    return True

def del_user(user_id: int) -> bool:
    return True

def add_channel(user_id: int, channel: str) -> bool:
    return True

def del_channel(user_id: int, channel: str) -> bool:
    return True

def predict(user_id: int, data: PredictedData) -> PredictedData:
    return data

def train(user_id: int, channel: str, text: str, label: bool) -> None:
    return
