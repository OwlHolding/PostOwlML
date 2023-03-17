import core.telegram as telegram
import asyncio


def test_channels():
    assert asyncio.run(telegram.get_posts("forbesrussia", 10, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("mintsifry", 10, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("SKOLKOVO_Start", 10, 0))[1] == 200


def test_count():
    assert asyncio.run(telegram.get_posts("forbesrussia", 100, 0))[1] == 200
    assert asyncio.run(telegram.get_posts("forbesrussia", 500, 0))[1] == 200


