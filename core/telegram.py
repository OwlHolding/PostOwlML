import threading
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from threading import Thread


class ChannelPool:

    def __init__(self):
        self.main = dict()

    def __getitem__(self, item):
        if not (item in self.main.keys()):
            self.main[item] = dict()

        return self.main


pool = ChannelPool()


class Worker(Thread):

    def __init__(self, channel, post_id):
        super().__init__()
        self.post_id = post_id
        self.channel = channel

    def run(self):
        pool[self.channel][self.post_id] = requests.get(f'https://t.me/{self.channel}/{self.post_id}?embed=1&mode=tme').text


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

    for i in range(int(post_id.split('/')[1]) - count + 1, int(post_id.split('/')[1]) + 1):
        worker = Worker(channel, i)
        worker.start()

    logging.debug(f"Started thread stack for {channel}")

    main_thread = threading.main_thread()

    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()

    logging.debug(f"Threading finished success")

    for page_id in range(int(post_id.split('/')[1]) - count + 1, int(post_id.split('/')[1]) + 1):
        html = BeautifulSoup(pool[channel][page_id], "html.parser")
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
                    break
            else:
                response.add(text)

    if len(response) == 0:
        return "", 400

    return list(response), 200
