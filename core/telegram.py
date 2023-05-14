import requests
from rss_parser import Parser
from threading import Thread
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from core.utils import get_random_agent


def get_index(channel_name: str, retry_count: int) -> int:
    """Получение id последнего поста в канале"""

    for i in range(retry_count):
        first_page = requests.get(f'https://t.me/s/{channel_name}/', headers=get_random_agent())
        soup = BeautifulSoup(first_page.text, "html.parser")
        if not (soup.find('div', class_='footer_telegram_description') or (soup.find('div', class_='tgme_page_description') and soup.find('div', class_='tgme_page_description').text.startswith('If you have Telegram, you can contact '))):
            break
    else:
        return -1

    last_post = soup.findAll('div', class_='tgme_widget_message_wrap js-widget_message_wrap')[-1]
    post_id = last_post.find('div', class_='tgme_widget_message text_not_supported_wrap js-widget_message')['data-post']

    post_id = post_id[post_id.rfind("/")+1:]

    return int(post_id)


def get_posts_rss(channel_name: str, count: int, times: [int, None]) -> tuple:
    """Чтение постов с канала с использованием RSS"""

    resp = requests.get(f"https://rsshub.app/telegram/channel/{channel_name}")
    parser = Parser(xml=resp.content)
    feed = parser.parse().feed

    if times:
        time_point = datetime(datetime.now().year, datetime.now().month, datetime.now().day, times // 60,
                              times % 60,
                              0) - timedelta(hours=24)

    response = set()
    i = 0
    while len(response) < count and i < len(feed):
        post = feed[i]

        if times:
            if datetime.strptime(post.publish_date, r"%a, %d %b %Y %H:%M:%S %Z").timestamp() > time_point.timestamp():
                response.add(post.description)
        else:
            response.add(post.description)

        i += 1

    return list(response), 200


def channel_is_exist(channel):
    """Проверка на канала"""
    resp = requests.get(f"https://rsshub.app/telegram/channel/{channel}")
    if resp.status_code != 200:
        return False
    return True


def get_html(url: str, results: list, index: int) -> None:
    """Получение html-страницы"""

    resp = requests.get(url, headers=get_random_agent())
    if resp.status_code == 200:
        results[index] = resp.text


def get_posts(channel_name: str, count: int, times: [int, None]) -> tuple:
    """Функция получения постов с канала """

    if times:
        time_point = datetime(datetime.now().year, datetime.now().month, datetime.now().day, times // 60,
                              times % 60,
                              0) - timedelta(hours=24)

    if not channel_is_exist(channel_name):
        return [], 400

    response = set()

    last_index = get_index(channel_name, 3)

    if last_index == -1:
        return get_posts_rss(channel_name, count, times)

    urls = [f'https://t.me/{channel_name}/{i}?embed=1&mode=tme' for i in range(last_index - count + 1, last_index + 1)]

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
        except:
            pass

        else:
            if times:
                if pub_time.timestamp() > time_point.timestamp():
                    response.add(text)
            else:
                response.add(text)

    return list(response), 200
