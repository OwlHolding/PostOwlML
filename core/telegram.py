import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from threading import Thread
import re
from core.utils import retry


def get_url(url, results, index):
    resp = requests.get(url)
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


@retry(times=3, exceptions=Exception)
async def get_posts(channel: str, count: int, times: [int, None]) -> [list[str], int]:
    """Запрос постов с телеграмм канала"""
    if times:
        time_point = datetime(datetime.now().year, datetime.now().month, datetime.now().day, times // 60,
                              times % 60,
                              0) - timedelta(hours=24)

    response = set()

    page = requests.get(f'https://t.me/s/{channel}/')
    soup = BeautifulSoup(page.text, "html.parser")

    last_post = soup.findAll('div', class_='tgme_widget_message_wrap js-widget_message_wrap')[-1]
    post_id = last_post.find('div', class_='tgme_widget_message text_not_supported_wrap js-widget_message')[
        'data-post']

    url_list = [f'https://t.me/{channel}/{i}?embed=1&mode=tme' for i in range(int(post_id.split('/')[1]) - count + 1, int(post_id.split('/')[1]) + 1)]

    result = download_html(url_list)

    for page in result:
        html = BeautifulSoup(page, "html.parser")
        try:
            text = re.sub('<div [^<]+?>', '', ''.join(
                map(str,
                    html.findAll('div', class_='tgme_widget_message_text js-message_text')[0].contents)).replace(
                '<br/>', '\n').replace("</div>", ""))
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
        return "", 200

    return list(response), 200

