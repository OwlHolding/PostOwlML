import torch
import wandb
import warnings
import numpy as np
import pandas as pd


from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from utils import MultiModalDataset
from models import ItemEmbeddingModel, UserEmbeddingModel, Decoder

warnings.simplefilter('ignore')


def train(user_embedding_model, item_embedding_model, criterion, optimizer, device,
          decoder_model, news, users, batch_size=5, images_rate=0.5, log_wandb=False, epochs=1):
    ue_model.to(device)
    ie_model.to(device)
    decoder.to(device)
    criterion.to(device)
    if log_wandb:
        wandb.login(key='5bcd93f54a66c38eeb79903c0ac633d5d3c3fab5')
        wandb.init(project='ZESRec', config=config)
    print('\n'*10)
    for epoch in range(epochs):
        stage = epoch+1
        mean_loss = []
        mean_f1 = []
        dataset = MultiModalDataset(
            news=news,
            users=users,
            stage=stage,
            images_rate=images_rate,
        )
        loader = DataLoader(dataset, batch_size=batch_size)
        with tqdm(total=len(loader)) as progress_bar:
            for batch in loader:
                news_u, news_i, labels = batch
                news_i = torch.stack(news_i).to(device)
                news_u = torch.stack(news_u).to(device)

                i_embeddings = item_embedding_model(news_u).resize(batch_size, stage, 300)

                u_embedding = user_embedding_model(i_embeddings).resize(batch_size, 1, 300)

                i_embeddings = item_embedding_model(news_i).resize(batch_size, stage, 300)
                output = torch.cat(decoder_model(u_embedding, i_embeddings)).resize(batch_size*stage)
                labels = labels.squeeze(0).to(device).resize(batch_size*stage)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                mean_loss.append(loss.item())
                f1 = f1_score(
                    np.around(labels.cpu().detach().numpy()),
                    np.around(output.cpu().detach().numpy())
                )
                mean_f1.append(f1)
                if log_wandb:
                    wandb.log({'Loss': loss.item(), 'F1': f1, 'Mean F1': sum(mean_f1)/ len(mean_f1), 'Mean Loss':sum(mean_loss)/ len(mean_loss) })
                progress_bar.set_description(f'Loss: {sum(mean_loss)/ len(mean_loss)}, F1: {sum(mean_f1)/ len(mean_f1)}')
                progress_bar.update()


if __name__ == '__main__':
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'embedding_size': 300,
        'dropout_rate': 0.2,
        'encoder_size': 768,
        'gru_input_size': 300,
        'gru_hidden_size': 300,
        'gru_num_layers': 2,
        'images_rate': 0.5
    }

    ie_model = ItemEmbeddingModel(
        embedding_size=300,
        encoder_size=768,
        intermediate_sizes=[512, 300],
        dropout_rate=0.2,
    )

    ue_model = UserEmbeddingModel(
        gru_input_size=300,
        gru_hidden_size=300,
        gru_num_layers=1,
        dropout=0.2,
    )

    decoder = Decoder()

    train(
        news=pd.read_feather(r'datasets/MIND/news_train.feather'),
        users=pd.read_feather(r'datasets/MIND/users_train.feather'),
        user_embedding_model=ue_model,
        item_embedding_model=ie_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(list(ue_model.parameters()) + list(ie_model.parameters())),
        device=torch.device(config['device']),
        decoder_model=decoder,
        epochs=20,
        log_wandb=True,
        batch_size=50,
    )
