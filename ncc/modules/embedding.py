import torch
import torch.nn as nn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    torch.manual_seed(1)
    m = nn.Embedding(num_embeddings, embedding_dim)  # , padding_idx=padding_idx
    # nn.init.uniform_(m.weight, -0.5, 0.5) # TODO
    # if padding_idx is not None:
    #     nn.init.constant_(m.weight[padding_idx], 0)
    return m
