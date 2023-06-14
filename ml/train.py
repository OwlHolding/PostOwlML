import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ItemEmbeddingModel, UserEmbeddingModel, Decoder
from utils import MultiModalDataset, seed_everything

warnings.simplefilter('ignore')


def validation(user_embedding_model, item_embedding_model, criterion, device,
               val_loader, decoder_model, log_wandb=False):
    predict_f1 = []
    predict_auc = []
    val_loss = []

    target_f1 = []
    target_auc = []
    item_embedding_model.eval()
    user_embedding_model.eval()
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as progress_bar:
            for batch in val_loader:
                news_u, news_i, labels = batch
                news_i = torch.stack(news_i).to(device)
                news_u = torch.stack(news_u).to(device)

                i_embeddings = item_embedding_model(news_u).permute(1, 0, 2)

                u_embedding = user_embedding_model(i_embeddings)

                i_embeddings = item_embedding_model(news_i).permute(1, 0, 2)

                output = decoder_model(u_embedding, i_embeddings).float()

                labels = torch.flatten(labels).float().to(device)

                loss = criterion(output, labels)
                val_loss.append(loss.item())
                predict_auc.append(output.detach().cpu().numpy())
                target_auc.append(labels.detach().cpu().numpy())

                predict_f1.append(np.around(output.detach().cpu().numpy()))
                target_f1.append(np.around(labels.detach().cpu().numpy()))

                progress_bar.set_description(f'Loss: {np.mean(val_loss)}')
                progress_bar.update()
        if log_wandb:
            wandb.log({
                'Validation F1': f1_score(y_true=np.concatenate(target_f1), y_pred=np.concatenate(predict_f1)),
                'Validation Roc Auc': roc_auc_score(y_true=np.concatenate(target_auc), y_score=np.concatenate(predict_auc)),
                'Validation Loss': np.mean(val_loss)
            })


def train(user_embedding_model, item_embedding_model, criterion, optimizer, device, config,
          decoder_model, news, users, val_news, val_users, checkpoint_dir, batch_size, images_rate=0.5, log_wandb=False, epochs=100,
          seed=42):
    seed_everything(seed)
    ue_model.to(device)
    ie_model.to(device)
    decoder.to(device)
    criterion.to(device)
    if log_wandb:
        wandb.login(key='5bcd93f54a66c38eeb79903c0ac633d5d3c3fab5')
        wandb.init(project='ZESRec', config=config)
    length = 101
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        mean_loss = []
        mean_roc_auc = []
        mean_f1 = []
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
        with tqdm(total=len(loader)) as progress_bar:
            for batch in loader:
                news_u, news_i, labels = batch
                news_i = torch.stack(news_i).to(device)
                news_u = torch.stack(news_u).to(device)

                i_embeddings = item_embedding_model(news_u).permute(1, 0, 2)

                u_embedding = user_embedding_model(i_embeddings)

                i_embeddings = item_embedding_model(news_i).permute(1, 0, 2)

                output = decoder_model(u_embedding, i_embeddings)

                labels = torch.flatten(labels).to(device)

                loss = criterion(output.float(), labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                mean_loss.append(loss.item())

                roc_auc = roc_auc_score(
                    y_true=labels.detach().cpu().numpy(),
                    y_score=output.detach().cpu().numpy(),
                )
                f1 = f1_score(
                    y_true=np.around(labels.detach().cpu().numpy()),
                    y_pred=np.around(output.detach().cpu().numpy()),
                )
                mean_f1.append(f1)
                mean_roc_auc.append(roc_auc)
                if log_wandb:
                    wandb.log({'Loss': loss.item(), 'Roc Auc': roc_auc, 'Mean Roc Auc': np.mean(mean_roc_auc),
                               'Mean Loss': np.mean(mean_loss), 'F1': f1, 'Mean F1': np.mean(mean_f1)})
                progress_bar.set_description(
                    f'Loss: {np.mean(mean_loss)},  Roc Auc: {np.mean(mean_roc_auc)}')
                progress_bar.update()

            torch.save(user_embedding_model.state_dict(), checkpoint_dir / f'ue_model_{epoch}.pt')
            torch.save(item_embedding_model.state_dict(), checkpoint_dir / f'ie_model_{epoch}.pt')
            progress_bar.set_description('Models saved', refresh=True)
            validation(user_embedding_model, item_embedding_model, nn.BCELoss(), device,
                       val_loader, decoder_model, log_wandb=log_wandb)
        length -= 1


if __name__ == '__main__':
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'embedding_size': 256,
        'dropout_rate': 0.2,
        'encoder_size': 512,
        'gru_num_layers': 2,
        'images_rate': 0.5,
        'lr': 0.00001,
        'epochs': 100,
        'batch_size': 128,
        'log_wandb': True,
    }

    ie_model = ItemEmbeddingModel(
        embedding_size=config['embedding_size'],
        encoder_size=config['encoder_size'],
        intermediate_sizes=[512, 256],
        dropout_rate=config['dropout_rate'],
    )

    ue_model = UserEmbeddingModel(
        gru_input_size=config['embedding_size'],
        gru_hidden_size=config['embedding_size'],
        gru_num_layers=config['gru_num_layers'],
        dropout=config['dropout_rate'],
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
        criterion=nn.BCELoss(),
        optimizer=optim.AdamW(list(ue_model.parameters()) + list(ie_model.parameters()), lr=config['lr']),
        device=torch.device(config['device']),
        decoder_model=decoder,
        epochs=config['epochs'],
        log_wandb=config['log_wandb'],
        batch_size=config['batch_size'],
        checkpoint_dir=Path('checkpoints') / 'from_max',
        config=config,
        val_news=mind_val,
        val_users=pd.read_feather(r'datasets/MIND/users_val.feather'),
    )
