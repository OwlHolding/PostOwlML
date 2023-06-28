from .ie_model import ItemEmbeddingModel
from .ue_model import UserEmbeddingModel
from .decoder_model import Decoder
import torch

config_ml = {
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