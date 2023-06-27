import warnings
from pathlib import Path

import os
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ItemEmbeddingModel, UserEmbeddingModel, Decoder, config_ml
from utils import MultiModalDataset, seed_everything

warnings.simplefilter('ignore')


def validation(user_embedding_model, item_embedding_model, criterion, device,
               val_loader, decoder_model, log_wandb=False):
    f1_pred = []
    roc_auc_pred = []
    val_loss = []

    f1_true = []
    roc_auc_true = []
    item_embedding_model.eval()
    user_embedding_model.eval()
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as progress_bar:
            for batch in val_loader:
                news_u, news_i, labels = batch
                news_i = torch.stack(news_i).to(device)
                news_u = torch.stack(news_u).to(device)

                i_embeddings = item_embedding_model(news_u).permute(1, 0, 2)

                u_embedding = user_embedding_model(i_embeddings)[-1, :, :]

                i_embeddings = item_embedding_model(news_i).permute(1, 0, 2)

                output = torch.cat(decoder_model(u_embedding, i_embeddings))


                labels = torch.flatten(labels).float().to(device)

                loss = criterion(output, labels.long())
                val_loss.append(loss.item())
                roc_auc_pred.append(output.detach().cpu().numpy())
                roc_auc_true.append(labels.detach().cpu().numpy())

                f1_pred.append(output.detach().cpu().numpy())
                f1_true.append(labels.detach().cpu().numpy())

                progress_bar.set_description(f'Loss: {np.mean(val_loss)}')
                progress_bar.update()
        if log_wandb:
            wandb.log({
                'Validation F1': f1_score(y_true=np.around(np.concatenate(f1_true)), y_pred=np.around(np.concatenate(f1_pred))),
                'Validation Roc Auc': roc_auc_score(y_true=np.concatenate(roc_auc_true),
                                                    y_score=np.concatenate(roc_auc_pred)),
                'Validation Loss': np.mean(val_loss)
            })


def train(user_embedding_model, item_embedding_model, criterion, optimizer, device, config_zesrec,
          decoder_model, news, users, val_news, val_users, checkpoint_dir, batch_size, images_rate=0.5, log_wandb=False,
          epochs=100,
          seed=42):
    seed_everything(seed)
    ue_model.to(device)
    ie_model.to(device)
    decoder.to(device)
    criterion.to(device)
    if log_wandb:
        wandb.login(key='5bcd93f54a66c38eeb79903c0ac633d5d3c3fab5')
        wandb.init(project='ZESRec', config=config_zesrec)
    length = 101
    for epoch in range(epochs):
        print(f'Epoch {epoch}, Length {length}')
        mean_loss = []
        dataset = MultiModalDataset(
            news=news,
            users=users,
            length=length,
            images_rate=images_rate,
        )
        val_dataset = MultiModalDataset(
            news=val_news,
            users=val_users,
            length=length,
            images_rate=images_rate,
        )
        item_embedding_model.train()
        user_embedding_model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        roc_auc_pred = []
        roc_auc_true = []
        f1_pred = []
        f1_true = []
        with tqdm(total=len(loader)) as progress_bar:
            for batch in loader:
                news_u, news_i, labels = batch
                news_i = torch.stack(news_i).to(device)
                news_u = torch.stack(news_u).to(device)

                i_embeddings = item_embedding_model(news_u).permute(1, 0, 2)

                u_embedding = user_embedding_model(i_embeddings)[-1, :, :]

                i_embeddings = item_embedding_model(news_i).permute(1, 0, 2)

                output = torch.cat(decoder_model(u_embedding, i_embeddings), dim=0)

                labels = torch.flatten(labels).to(device)

                loss = criterion(output.float(), labels.long())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                mean_loss.append(loss.item())
                roc_auc_pred.append(output.detach().cpu().numpy())
                roc_auc_true.append(labels.detach().cpu().numpy())
                f1_pred.append(output.detach().cpu().numpy())
                f1_true.append(labels.detach().cpu().numpy())

                if log_wandb:
                    wandb.log({'Loss': loss.item(), 'Mean Loss': np.mean(mean_loss)})
                progress_bar.set_description(
                    f'Loss: {np.mean(mean_loss)}')
                progress_bar.update()

            torch.save(user_embedding_model.state_dict(), checkpoint_dir / f'ue_model_{epoch}.pt')
            torch.save(item_embedding_model.state_dict(), checkpoint_dir / f'ie_model_{epoch}.pt')
            progress_bar.set_description('Models saved', refresh=True)
            validation(user_embedding_model, item_embedding_model, criterion, device,
                       val_loader, decoder_model, log_wandb=log_wandb)
            if log_wandb:
                wandb.log({
                    'Train F1': f1_score(y_true=np.around(np.concatenate(f1_true)),
                                         y_pred=np.around(np.concatenate(f1_pred))),
                    'Train Roc Auc': roc_auc_score(y_true=np.concatenate(roc_auc_true),
                                                   y_score=np.concatenate(roc_auc_pred))
                })

            length -= 1


if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'online'
    ie_model = ItemEmbeddingModel(
        embedding_size=config_ml['embedding_size'],
        encoder_size=config_ml['encoder_size'],
        intermediate_sizes=[512, 256],
        dropout_rate=config_ml['dropout_rate'],
    )

    ue_model = UserEmbeddingModel(
        gru_input_size=config_ml['embedding_size'],
        gru_hidden_size=config_ml['embedding_size'],
        gru_num_layers=config_ml['gru_num_layers'],
        dropout=config_ml['dropout_rate'],
    )

    decoder = Decoder()
    mind = pd.read_feather(r'datasets/MIND/news_train.feather')
    mind.set_index('NewsID', inplace=True)
    mind_val = pd.read_feather(r'datasets/MIND/news_val.feather')
    mind_val.set_index('NewsID', inplace=True)

    train(
        news=mind,
        users=pd.read_feather(r'datasets/MIND/users_train.feather'),
        user_embedding_model=ue_model,
        item_embedding_model=ie_model,
        criterion=nn.NLLLoss(),
        optimizer=optim.AdamW(list(ue_model.parameters()) + list(ie_model.parameters()), lr=config_ml['lr']),
        device=torch.device(config_ml['device']),
        decoder_model=decoder,
        epochs=config_ml['epochs'],
        log_wandb=config_ml['log_wandb'],
        batch_size=config_ml['batch_size'],
        checkpoint_dir=Path('checkpoints') / '3',
        config_zesrec=config_ml,
        val_news=mind_val,
        val_users=pd.read_feather(r'datasets/MIND/users_val.feather'),
    )
