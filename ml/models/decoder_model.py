import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_embedding, item_embedding):
        return F.softmax(torch.matmul(user_embedding, item_embedding.T), dim=-1)