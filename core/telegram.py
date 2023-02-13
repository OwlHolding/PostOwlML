import requests
from datetime import datetime, timedelta
import json


async def get_posts(channel: str, count: int, times: [int, None]) -> [list[str], int]:
    """Запрос постов с телеграмм канала"""

    url = "https://telegram92.p.rapidapi.com/api/history/channel"

    querystring = {"channel": channel, "limit": str(count), "offset": "0"}

    headers = {
        "X-RapidAPI-Key": "aa3ab5fdc9mshb8cb5b9c9c4bacfp1190ecjsnc6db34f10782",
        "X-RapidAPI-Host": "telegram92.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    if 'error' in response.text:
        return [], 400

    response = json.loads(response.text)['messages']

    posts = []

    if times:
        time_point = datetime(datetime.now().year, datetime.now().month, datetime.now().day, times // 60, times % 60,
                              0) - timedelta(hours=24)
        for post in response:
            if datetime.utcfromtimestamp(post['date']) > time_point:
                posts.append(post['text'])
    else:
        for post in response:
            posts.append(post['text'])

    return posts, 201
