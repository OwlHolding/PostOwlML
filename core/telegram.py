import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import aiohttp
import asyncio


async def get_one_post(session, url):
    async with session.get(url) as response:
        text = await response.text()
        return text, url


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

    url_list = [f'https://t.me/{channel}/{i}?embed=1&mode=tme' for i in
                range(int(post_id.split('/')[1]) - count + 1, int(post_id.split('/')[1]) + 1)]

    tasks = []
    headers = {
        "user-agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        for url in url_list:
            tasks.append(get_one_post(session, url))

        htmls = await asyncio.gather(*tasks)

    pages = []

    for html in htmls:
        if html is not None:
             pages.append(html[0])
        else:
            continue

    for page in pages:
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
                    break
            else:
                response.add(text)

    if len(response) == 0:
        return "", 400

    return list(response), 200
