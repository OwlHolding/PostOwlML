import core.telegram as telegram
import asyncio
from starlette.testclient import TestClient
from app import app

client = TestClient(app)


def test_telegram_channels():
    assert asyncio.run(telegram.get_posts("forbesrussia", 10, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("mintsifry", 10, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("SKOLKOVO_Start", 10, 0))[1] == 200


def test_telegram_count():
    assert asyncio.run(telegram.get_posts("forbesrussia", 100, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("forbesrussia", 500, 0))[1] == 200


def test_ping():
    response = client.post('/register/1')
    assert response.status_code // 10 == 20


def test_register_channel():
    response = client.post('/regchannel/1/forbesrussia')
    assert response.status_code // 10 == 20

