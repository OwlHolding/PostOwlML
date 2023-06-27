from starlette.testclient import TestClient
from app import app
from ml import train, predict

client = TestClient(app)


def test_add_user():
    response = client.post("/add-user/1/")
    assert response.status_code // 10 == 20


def test_del_user():
    response = client.post("/del-user/1/")
    assert response.status_code // 10 == 20


def test_add_channel():
    response = client.post("/add-channel/1/forbesrussia")
    assert response.status_code // 10 == 20


def test_del_channel():
    response = client.post("/del-channel/1/forbesrussia")
    assert response.status_code // 10 == 20


def test_train():
    train(1, 'forbes', post='''Forbes USA изучил деятельность Stability AI, компании с оборотом $1 млрд, и ее основателя
         Эмада Мостака. 
    Обладатель степени магистра Оксфордского университета, Мостак — титулованный менеджер хедж-фондов, доверенное лицо
     Организации Объединенных Наций и создатель технологии Stable Diffusion.
    Сегодня бизнесмена можно назвать одной из главных фигур волны генеративного ИИ.
     Кроме того, своей компании он обеспечил свыше $100 млн на реализацию собственного представления о том,
      каким нужно строить по-настоящему открытый ИИ. «Надеюсь, за такое мне положена Нобелевская премия», - шутил он в
       январском интервью для Forbes.
    По крайней мере, все это рассказывает он сам.
    На самом же деле в Оксфорде Мостак получил степень бакалавра, а не магистра. Это не единственная ложь, которую
     рассказывает Мостак, чтобы пробиться в авангард движения.
    Редакция провела интервью с 13 бывшими и нынешними сотрудниками компании и более чем двумя десятками инвесторов, а также
     проанализировала презентации и внутренние документы.
    Как показало расследование, успех стартапа во многом обеспечили приписывание себе чужих успехов и откровенная ложь его 
    гендиректора. 
    Как основатель Stability AI добился успеха благодаря лжи - читайте на сайте Forbes (https://www.forbes.ru/investicii/490
    540-mnogoe-ne-shoditsa-kak-osnovatel-stability-ai-dobilsa-uspeha-blagodara-lzi)''', label=True)


def test_predict():
    assert len(predict(post='''Из-за разрушения Каховской ГЭС погибли 
        (https://www.forbes.ru/society/491068-glava-hersonskoj-oblasti-nazval-cislo-pogibsih-iz-za-razrusenia-kahovskoj-ges?
        utm_source=forbes&utm_campaign=lnews)
         25 человек, еще 17 человек числятся пропавшими без вести, заявил врио губернатора Херсонской области Владимир
          Сальдо. С территорий ниже ГЭС по течению
          Днепра эвакуированы около 8000 человек''', users=[1, 2], channel='forbes')) == 2
