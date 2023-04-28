import requests
from rss_parser import Parser
from threading import Thread
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging

from core.utils import get_random_agent


def get_index(channel_name: str) -> int:
    """Получение индекса первого поста"""

    page = requests.get(f"https://rsshub.app/telegram/channel/{channel_name}", headers=get_random_agent()).text
    parser = Parser(xml=page)
    feed = parser.parse()

    link = feed.feed[-1].link
    index = link[link.rfind("/")+1:]

    return int(index)


def get_html(url: str, results: list, index: int) -> None:
    """Получение html-страницы"""

    resp = requests.get(url, headers=get_random_agent())
    if resp.status_code == 200:
        results[index] = resp.text


def get_posts(channel_name: str, count: int, times: [int, None]):
    """Функция получения постов с канала """

    if times:
        time_point = datetime(datetime.now().year, datetime.now().month, datetime.now().day, times // 60,
                              times % 60,
                              0) - timedelta(hours=24)

    response = set()

    last_index = get_index(channel_name)

    urls = [f'https://t.me/{channel_name}/{i}?embed=1&mode=tme' for i in range(last_index - count - 1, last_index + 1)]

    threads = [None] * count
    results = [None] * count

    for i in range(len(threads)):
        threads[i] = Thread(target=get_html, args=[urls[i], results, i])
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()

    for page in results:
        try:
            html = BeautifulSoup(page, "html.parser")
            text = ''.join(
                map(str,
                    html.findAll('div', class_='tgme_widget_message_text js-message_text')[0].contents)).replace(
                '<br/>', '\n')
            pub_time = datetime.strptime(html.find('time')['datetime'], "%Y-%m-%dT%H:%M:%S%z")

        except Exception as e:
            logging.error(e)

        else:
            if times:
                if pub_time.timestamp() > time_point.timestamp():
                    response.add(text)
            else:
                response.add(text)

    if len(response) == 0:
        return [""], 200

    return list(response), 200
