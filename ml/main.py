from pathlib import Path
import torch
import pymongo
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import bson
from .models import ItemEmbeddingModel, UserEmbeddingModel, Decoder, config
import pickle
import logging

MONGO_URL = 'mongodb://localhost:27017/'


def add_user(user_id: int) -> bool:
    try:
        client = pymongo.MongoClient(host=MONGO_URL)
        db = client.postowl
        coll = db.users
        coll.insert_one({'_id': user_id, 'channels': [], 'embedding': None})
        return True
    except Exception as e:
        return False


def del_user(user_id: int) -> bool:
    try:
        client = pymongo.MongoClient(host=MONGO_URL)
        db = client.postowl
        coll = db.users

        coll.delete_one({'_id': user_id})
        return True
    except Exception as e:
        return False


def add_channel(user_id: int, channel: str) -> bool:
    try:
        client = pymongo.MongoClient(host=MONGO_URL)
        db = client.postowl
        coll = db.users

        channels = coll.find_one({'_id': user_id})['channels']
        channels.append({'_id': channel})
        coll.update_one({'_id': user_id},
                        {'$set': {'channels': channels}
                         })
        return True
    except Exception as e:
        logging.info('ML: User already registered')
        return False


def del_channel(user_id: int, channel: str) -> bool:
    try:
        client = pymongo.MongoClient(host=MONGO_URL)
        db = client.postowl
        coll = db.users
        channels = [i for i in coll.find_one({'_id': user_id})['channels'] if i['_id'] != channel]
        coll.update_one({'_id': user_id},
                        {'$set': {'channels': channels}
                         })
        return True
    except Exception as e:
        logging.info(e)
        return False


def predict(post: str, channel: str, users: list[int]) -> list[int]:
    with torch.no_grad():
        device = torch.device(config['device'])
        try:
            client = pymongo.MongoClient(host=MONGO_URL)
            db = client.postowl
            coll = db.users
        except Exception as e:
            logging.error(f'ML: {e}')
            return users
        text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

        item_embedding_model = ItemEmbeddingModel(
            embedding_size=config['embedding_size'],
            encoder_size=config['encoder_size'],
            intermediate_sizes=[512, 256],
            dropout_rate=config['dropout_rate'],
        )

        user_embedding_model = UserEmbeddingModel(
            gru_input_size=config['embedding_size'],
            gru_hidden_size=config['embedding_size'],
            gru_num_layers=config['gru_num_layers'],
            dropout=config['dropout_rate'],
        )

        decoder = Decoder()
        item_embedding_model.load_state_dict(torch.load(Path('ie_model.pt')))
        item_embedding_model.eval()
        item_embedding_model.to(device)
        user_embedding_model.to(device)

        predict = []
        post = text_encoder.encode(post, batch_size=config['batch_size'], device=config['device'], convert_to_tensor=True)
        item_embedding = item_embedding_model(post)
        for user_id in users:
            try:
                user_embedding = coll.find_one({'_id': user_id})['embedding']
            except Exception as e:
                user_embedding = None
                logging.error(f'ML: {e}')
            if user_embedding is None:
                predict.append(user_id)
                continue
            user_embedding = torch.from_numpy(pickle.loads(user_embedding)).to(device)
            pred = decoder([item_embedding], [user_embedding])[0].cpu().numpy()
            if pred > 0.5:
                predict.append(user_id)


    return predict


def train(user_id: int, channel: str, post: str, label: bool) -> None:
    if label:
        with torch.no_grad():
            try:
                client = pymongo.MongoClient(host=MONGO_URL)
                db = client.postowl
                coll = db.users
                user_embedding = coll.find_one({'_id': user_id})['embedding']
            except Exception as e:
                logging.error(f"ML: {e}")
                return
            device = torch.device(config['device'])
            text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/clip-ViT-B-32-multilingual-v1')
            item_embedding_model = ItemEmbeddingModel(
                embedding_size=config['embedding_size'],
                encoder_size=config['encoder_size'],
                intermediate_sizes=[512, 256],
                dropout_rate=config['dropout_rate'],
            )
            user_embedding_model = UserEmbeddingModel(
                gru_input_size=config['embedding_size'],
                gru_hidden_size=config['embedding_size'],
                gru_num_layers=config['gru_num_layers'],
                dropout=config['dropout_rate'],
            )
            user_embedding_model.load_state_dict(torch.load(Path('ue_model.pt')))
            item_embedding_model.load_state_dict(torch.load(Path('ie_model.pt')))
            item_embedding_model.eval()
            user_embedding_model.eval()
            item_embedding_model.to(device)
            user_embedding_model.to(device)
            text_encoder.to(device)


            if user_embedding is not None:
                user_embedding = torch.from_numpy(pickle.loads(user_embedding))
            post = tokenizer(
                post,
                max_length= 128,
                truncation= True,
                return_token_type_ids= False,
                padding= 'max_length',
                return_tensors= 'pt'
            )
            post = text_encoder({'input_ids': post['input_ids'].to(device),
                                'attention_mask': post['attention_mask'].to(device)})[
                'sentence_embedding']
            item_embedding = item_embedding_model(post).unsqueeze(0)

            user_embedding = user_embedding_model(item_embedding, user_embedding)[-1, :]
            try:
                coll.update_one({'_id': user_id},
                                {'$set': {'embedding': bson.Binary(pickle.dumps(user_embedding.cpu().numpy(), protocol=2))}
                                 })
            except Exception as e:
                logging.error(f"ML: {e}")
    return


if __name__ == '__main__':
    add_user(1)
    add_user(2)
    train(1, 'forbes', post='''Forbes USA изучил деятельность Stability AI, компании с оборотом $1 млрд, и ее основателя Эмада Мостака. 
Обладатель степени магистра Оксфордского университета, Мостак — титулованный менеджер хедж-фондов, доверенное лицо Организации Объединенных Наций и создатель технологии Stable Diffusion.
Сегодня бизнесмена можно назвать одной из главных фигур волны генеративного ИИ.
 Кроме того, своей компании он обеспечил свыше $100 млн на реализацию собственного представления о том,
  каким нужно строить по-настоящему открытый ИИ. «Надеюсь, за такое мне положена Нобелевская премия», - шутил он в январском интервью для Forbes.
По крайней мере, все это рассказывает он сам.
На самом же деле в Оксфорде Мостак получил степень бакалавра, а не магистра. Это не единственная ложь, которую рассказывает Мостак, чтобы пробиться в авангард движения.
Редакция провела интервью с 13 бывшими и нынешними сотрудниками компании и более чем двумя десятками инвесторов, а также проанализировала презентации и внутренние документы.
Как показало расследование, успех стартапа во многом обеспечили приписывание себе чужих успехов и откровенная ложь его гендиректора. 
Как основатель Stability AI добился успеха благодаря лжи - читайте на сайте Forbes (https://www.forbes.ru/investicii/490540-mnogoe-ne-shoditsa-kak-osnovatel-stability-ai-dobilsa-uspeha-blagodara-lzi)''', label=True)
    print(predict(post='''Из-за разрушения Каховской ГЭС погибли 
    (https://www.forbes.ru/society/491068-glava-hersonskoj-oblasti-nazval-cislo-pogibsih-iz-za-razrusenia-kahovskoj-ges?utm_source=forbes&utm_campaign=lnews)
     25 человек, еще 17 человек числятся пропавшими без вести, заявил врио губернатора Херсонской области Владимир Сальдо. С территорий ниже ГЭС по течению
      Днепра эвакуированы около 8000 человек''', users=[1, 2], channel='forbes'))
    del_user(1)
    del_user(2)
