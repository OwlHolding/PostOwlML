from starlette.testclient import TestClient
from app import app

client = TestClient(app)


def test_add_user():
    response = client.post("/add-user/1/")
    assert response.status_code // 10 == 20


def test_del_user():
    response = client.delete("/del-user/1/")
    assert response.status_code // 10 == 20


def test_add_channel():
    response = client.post("/add-channel/1/forbesrussia")
    assert response.status_code // 10 == 20


def test_del_channel():
    response = client.delete("/del-channel/1/forbesrussia")
    assert response.status_code // 10 == 20
