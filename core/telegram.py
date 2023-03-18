import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from threading import Thread
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

from core.utils import retry


def get_url(url, results, index):
    resp = requests.get(url, headers=get_random_agent())
    if resp.status_code == 200:
        results[index] = resp.text


def download_html(urls):
    threads = [None] * len(urls)
    results = [None] * len(urls)

    for i in range(len(threads)):
        threads[i] = Thread(target=get_url, args=[urls[i], results, i])
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()

    return results


def get_random_agent() -> dict:
    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]
    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

    user_agent = user_agent_rotator.get_random_user_agent()

    headers = {
        'User-Agent': user_agent,
    }

    return headers


@retry(times=30, exceptions=IndexError, time_sleep=5)
def get_index(channel: str) -> str:
    first_page = requests.get(f'https://t.me/s/{channel}/', headers=get_random_agent())
    soup = BeautifulSoup("<div>"+first_page.text+"", "html.parser")

    last_post = soup.findAll('div', class_='tgme_widget_message_wrap js-widget_message_wrap')[-1]
    post_id = last_post.find('div', class_='tgme_widget_message text_not_supported_wrap js-widget_message')['data-post']

    return post_id


@retry(times=5, exceptions=IndexError, time_sleep=5)
async def get_posts(channel: str, count: int, times: [int, None]) -> [list[str], int]:
    """Запрос постов с телеграмм канала"""
    if times:
        time_point = datetime(datetime.now().year, datetime.now().month, datetime.now().day, times // 60,
                              times % 60,
                              0) - timedelta(hours=24)

    response = set()

    post_id = get_index(channel)

    url_list = [f'https://t.me/{channel}/{i}?embed=1&mode=tme' for i in range(int(post_id.split('/')[1]) - count + 1, int(post_id.split('/')[1]) + 1)]

    result = download_html(url_list)

    for page in result:
        html = BeautifulSoup(page, "html.parser")
        try:
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

    if len(response) == 0:
        return [""], 200

    return list(response), 200

