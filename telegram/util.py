from random_user_agent.params import SoftwareName, OperatingSystem
from random_user_agent.user_agent import UserAgent


def remove_tags(text: str) -> str:
    """Удаление нечитаемых тегов"""
    open_ = -1
    opened = []
    buff = ""
    allow = {"b", "strong", "i", "em", "u",
             "ins", "s", "strike", "del", "span",
             "tg-spoiler", "b", "code", "pre", "a"}

    for i in range(len(text)):
        if open_ != -1:
            if text[i] == ' ' or text[i] == '>':
                if buff not in allow:
                    opened.append(open_)
                buff = ""
                open_ = -1
            else:
                buff += text[i] + ">"
        else:
            if text[i] == '<':
                open_ = i

    iter_ = 0
    bad = False
    result = ""

    for i in range(len(text)):
        if iter_ < len(opened) and opened[iter_] == i:
            bad = True
            iter_ += 1
        if text[i] == '>':
            bad = False
            continue
        if not bad:
            result += text[i]

    return result


def get_random_agent() -> dict:
    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]
    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

    user_agent = user_agent_rotator.get_random_user_agent()

    headers = {
        'User-Agent': user_agent,
    }

    return headers