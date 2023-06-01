class PredictedData:
    """Класс для хранения постов, по каналам и формирования ленты на основе хранимых данных"""

    def __init__(self, data: dict):
        """
        :param data:{
            "channel-name": ["text"]
        }"""
        self.data = data

    def get_feed(self) -> list[str]:
        feed = []
        for key in self.data.keys():
            feed += [text + f" \n <a href='t.me/{key}</a>" for text in self.data[key]]

        return feed

