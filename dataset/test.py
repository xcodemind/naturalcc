# -*- coding: utf-8 -*-


import torch


def position_tensor(k: int, max_len: int) -> torch.Tensor:
    '''
    References:
        Self-Attention with Relative Position Representations
        https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a

    build a position list, as follow
    [...(all are 0), 0, 1, ..., 2k, ...(all are 2k)]
    move a window on this list from right to left

    Examples:
        k=3, max_len=10
    Returns:
        tensor([[3, 4, 5, 6, 6, 6, 6, 6, 6, 6],
            [2, 3, 4, 5, 6, 6, 6, 6, 6, 6],
            [1, 2, 3, 4, 5, 6, 6, 6, 6, 6],
            [0, 1, 2, 3, 4, 5, 6, 6, 6, 6],
            [0, 0, 1, 2, 3, 4, 5, 6, 6, 6],
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 6],
            [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
            [0, 0, 0, 0, 0, 0, 0, 1, 2, 3]])
    '''
    MIN_POS, MAX_POS = 0, 2 * k
    base = [MIN_POS] * (max_len - k - 1) + list(range(2 * k + 1)) + [MAX_POS] * (max_len - k - 1)
    pos_tensor = [
        base[max_len - 1 - i:max_len - 1 - i + max_len]
        for i in range(max_len)
    ]
    pos_tensor = torch.Tensor(pos_tensor).long()
    return pos_tensor


def relative_position_embedding(query: torch.Tensor, key: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    '''
    References:
        https://github.com/tensorflow/tensor2tensor/blob/9e0a894034d8090892c238df1bd9bd3180c2b9a3/tensor2tensor/layers/common_attention.py#L1556-L1587
    Args:
        query: [batch_size, head, seq_len, dim]
        key: [batch_size, head, seq_len, dim]
        pos_embed: [seq_len, seq_len, dim]

    Returns:

    '''
    batch_size, head, seq_len, dim = query.size()
    # Q * K^T
    QK_T = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, head, seq_len, seq_len]

    # Q * A^T
    # [batch_size, head, seq_len, seq_len] =>  [seq_len, batch_size, head, seq_len] => [seq_len, batch_size*head, seq_len]
    query_T = query.permute(2, 0, 1, 3).view(seq_len, batch_size * head, dim)
    QA_T = torch.matmul(query_T, pos_emb.transpose(-1, -2))
    QA_T = QA_T.view(seq_len, batch_size, head, seq_len).permute(1, 2, 0, 3)
    return QK_T + QA_T


BATCH_SIZE = 4
MAX_SEQ_LEN = 10
BORDER_SIZE = 3
HEAD_NUM = 3
EMBEDDING_DIM = 256

Q = torch.rand(size=(BATCH_SIZE, HEAD_NUM, MAX_SEQ_LEN, EMBEDDING_DIM))
K = torch.rand(size=(BATCH_SIZE, HEAD_NUM, MAX_SEQ_LEN, EMBEDDING_DIM))

pos_tensor = position_tensor(k=BORDER_SIZE, max_len=MAX_SEQ_LEN)

postion_embedding = torch.nn.Embedding(2 * BORDER_SIZE + 1, EMBEDDING_DIM)
pos_emb = postion_embedding(pos_tensor)
out = relative_position_embedding(Q, K, pos_emb)
print(out.size())
