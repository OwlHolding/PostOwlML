import requests
from datetime import datetime
import json


async def get_posts(channel: str, count: int, times: [int, None]) -> list[str]:
    """Запрос постов с телеграмм канала"""

    url = "https://telegram92.p.rapidapi.com/api/history/channel"

    querystring = {"channel": channel, "limit": str(count), "offset": "0"}

    headers = {
        "X-RapidAPI-Key": "aa3ab5fdc9mshb8cb5b9c9c4bacfp1190ecjsnc6db34f10782",
        "X-RapidAPI-Host": "telegram92.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    response = json.loads(response.text)['messages']

    posts = []

    if times:
        for post in response:
            t = datetime.utcfromtimestamp(post['date'])
            if t.hour * t.minute * t.second > times:
                posts.append(post['text'])
    else:
        for post in response:
            posts.append(post['text'])

    return posts
