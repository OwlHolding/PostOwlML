import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from .database import S3Database
from .models import ItemEmbeddingModel, UserEmbeddingModel, Decoder, config_ml
import pickle


with open(Path(__file__).resolve().parents[1]/'config.json') as f:
    config = json.load(f)
DB = S3Database(config)


def add_user(user_id: int) -> bool:
    DB.add_user(user_id)
    return True


def del_user(user_id: int) -> bool:
    DB.del_user(user_id)
    return True


def add_channel(user_id: int, channel: str) -> bool:
    DB.add_channel(user_id, channel)
    return True


def del_channel(user_id: int, channel: str) -> bool:
    DB.del_channel(user_id, channel)
    return True


def predict(post: str, channel: str, users: list[int]) -> list[int]:
    with torch.no_grad():
        device = torch.device(config_ml['device'])
        text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

        item_embedding_model = ItemEmbeddingModel(
            embedding_size=config_ml['embedding_size'],
            encoder_size=config_ml['encoder_size'],
            intermediate_sizes=[512, 256],
            dropout_rate=config_ml['dropout_rate'],
        )

        user_embedding_model = UserEmbeddingModel(
            gru_input_size=config_ml['embedding_size'],
            gru_hidden_size=config_ml['embedding_size'],
            gru_num_layers=config_ml['gru_num_layers'],
            dropout=config_ml['dropout_rate'],
        )

        decoder = Decoder()
        item_embedding_model.load_state_dict(torch.load(Path(__file__).resolve().parents[0]/'ie_model.pt'))

        user_embedding_model.load_state_dict(torch.load(Path(__file__).resolve().parents[0]/'ue_model.pt'))
        user_embedding_model.eval()
        item_embedding_model.eval()
        item_embedding_model.to(device)
        user_embedding_model.to(device)

        pred = []
        post = text_encoder.encode(post, batch_size=config_ml['batch_size'], device=config_ml['device'], convert_to_tensor=True)
        item_embedding = item_embedding_model(post)
        for user_id in users:
            user_embedding = DB.get_emb(user_id)
            if user_embedding is None:
                pred.append(user_id)
                continue
            user_embedding = torch.from_numpy(pickle.loads(user_embedding)).to(device)
            o = decoder([item_embedding], [user_embedding])[0].cpu().numpy()
            if o > 0.5:
                pred.append(o)

    return pred


def train(user_id: int, channel: str, post: str, label: bool) -> None:
    if label:
        with torch.no_grad():

            user_embedding = DB.get_emb(user_id)
            device = torch.device(config_ml['device'])
            text_encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/clip-ViT-B-32-multilingual-v1')
            item_embedding_model = ItemEmbeddingModel(
                embedding_size=config_ml['embedding_size'],
                encoder_size=config_ml['encoder_size'],
                intermediate_sizes=[512, 256],
                dropout_rate=config_ml['dropout_rate'],
            )
            user_embedding_model = UserEmbeddingModel(
                gru_input_size=config_ml['embedding_size'],
                gru_hidden_size=config_ml['embedding_size'],
                gru_num_layers=config_ml['gru_num_layers'],
                dropout=config_ml['dropout_rate'],
            )

            user_embedding_model.load_state_dict(torch.load(Path(__file__).resolve().parents[0]/'ue_model.pt'))
            item_embedding_model.load_state_dict(torch.load(Path(__file__).resolve().parents[0]/'ie_model.pt'))
            item_embedding_model.eval()
            user_embedding_model.eval()
            item_embedding_model.to(device)
            user_embedding_model.to(device)
            text_encoder.to(device)

            if user_embedding is not None:
                user_embedding = torch.from_numpy(pickle.loads(user_embedding))
            post = tokenizer(
                post,
                max_length=128,
                truncation=True,
                return_token_type_ids=False,
                padding='max_length',
                return_tensors='pt'
            )
            post = text_encoder({'input_ids': post['input_ids'].to(device),
                                'attention_mask': post['attention_mask'].to(device)})[
                'sentence_embedding']
            item_embedding = item_embedding_model(post).unsqueeze(0)

            user_embedding = user_embedding_model(item_embedding, user_embedding)[-1, :]
            DB.update_emb(user_id, user_embedding)

    return



