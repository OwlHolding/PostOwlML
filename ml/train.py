import torch
import wandb
import warnings
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch import nn
from torch import optim


from transformers import VideoMAEModel
from transformers import RobertaTokenizer, Data2VecTextModel
from transformers import AutoImageProcessor, Data2VecVisionModel
from transformers import Data2VecAudioModel

from utils import Identity, MultiModalDataset
from models import ItemEmbeddingModel, UserEmbeddingModel, Decoder

warnings.simplefilter('ignore')


def train(dataset, user_embedding_model, item_embedding_model, criterion, optimizer, device, decoder_model, log_wandb=False, epochs=1):
    if log_wandb:
        wandb.login(key='5bcd93f54a66c38eeb79903c0ac633d5d3c3fab5')
        wandb.init(project='ZESRec', config=config)
    for epoch in range(epochs):
        mean_loss = []
        mean_f1 = []
        with tqdm(total=len(dataset)) as tq:
            for i in range(len(dataset)):
                news_u, news_i, labels = dataset[i]
                i_embeddings = []
                for news in news_u:
                    i_embeddings.append(item_embedding_model(**news))
                u_embedding = user_embedding_model(i_embeddings)
                i_embeddings = []
                for news in news_i:
                    i_embeddings.append(item_embedding_model(**news))
                i_embeddings = torch.cat(i_embeddings, dim=0)
                output = decoder_model(u_embedding, i_embeddings)
                labels.to(device)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                mean_loss.append(loss.item())
                f1 = f1_score(
                    list(map(round, labels.cpu().detach().numpy())),
                    list(map(round, output.cpu().detach().numpy()))
                )
                mean_f1.append(f1)
                if log_wandb:
                    wandb.log({'Loss': loss.item(), 'F1': f1, 'Mean F1': sum(mean_f1)/ len(mean_f1) })
                tq.set_description(f'Loss: {sum(mean_loss)/ len(mean_loss)}, F1: {sum(mean_f1)/ len(mean_f1)}')
                tq.update()


if __name__ == '__main__':
    config = {
        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'device': 'cpu',
        'embedding_size': 300,
        'dropout_rate': 0.2,
        'encoder_size': 768,
        'gru_input_size': 300,
        'gru_hidden_size': 300,
        'gru_num_layers': 2,
        'images_rate': 0.5
    }


    device = torch.device(config['device'])
    ie_model = ItemEmbeddingModel(
        text_encoder=Data2VecTextModel.from_pretrained("facebook/data2vec-text-base"),
        image_encoder=Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base"),
        audio_encoder=Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h"),
        video_encoder=VideoMAEModel.from_pretrained("MCG-NJU/videomae-base"),
        device=device,
        embedding_size=300,
        encoder_size=768,
        intermediate_sizes=[512, 300],
        dropout_rate=0.2,
    )

    ie_model.text_encoder.pooler.dense = Identity()

    ue_model = UserEmbeddingModel(
        gru_input_size=300,
        gru_hidden_size=300,
        gru_num_layers=2,
        dropout=0.2,
    )
    multi_modal_dataset = MultiModalDataset(
        news=pd.read_feather('datasets/MIND/news_train.feather'),
        users=pd.read_feather('datasets/MIND/users_train.feather'),
        images_folder=Path('datasets/MIND/Images'),
        image_preprocessor=AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base"),
        tokenizer=RobertaTokenizer.from_pretrained("facebook/data2vec-text-base"),
        images_rate=0.5,
    )

    criterion = nn.CrossEntropyLoss()
    decoder = Decoder()
    ue_model.to(device)
    ie_model.to(device)
    criterion.to(device)
    decoder.to(device)
    optimizer = optim.AdamW(list(ue_model.parameters()) + list(ie_model.parameters()))

    train(
        dataset=multi_modal_dataset,
        user_embedding_model=ue_model,
        item_embedding_model=ie_model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        decoder_model=decoder,
        epochs=20
    )
