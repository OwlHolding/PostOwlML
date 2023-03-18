import telebot

bot = telebot.TeleBot("6013065416:AAEity5Qy-SIDdcPF9G3nNynP9FBV48hXnI")

white_list = [2009584602]


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, f'Hello {message.chat.id}')


@bot.message_handler(commands=['log'])
def get_log(message):
    if message.chat.id in white_list:
        with open("log.txt", "r") as file:
            log = file.read().split('\n')[::-1]
        try:
            log = log[:10]
        except IndexError:
            pass

        bot.send_message(message.chat.id, "\n".join(log))
