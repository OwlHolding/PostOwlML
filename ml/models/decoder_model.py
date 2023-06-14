import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_embedding, item_embedding):
        res = []
        for u_e, i_e in zip(user_embedding, item_embedding):
            res.append(F.softmax(torch.matmul(u_e, i_e.T), dim=-1))
        return torch.cat(res, dim=0)