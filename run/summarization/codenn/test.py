import torch
from torch import nn
import torch.nn.functional as F


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class DotSimiAttn(nn.Module):
    """
    Dot similarity Attention

    """

    def __init__(self, hidden_size):
        super(DotSimiAttn, self).__init__()
        self.hidden_size = hidden_size

    # def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs):
    def forward(self, h, enc_out, enc_padding_mask, batchsize):
        # et = self.W_h(h)  # bs,n_seq, n_hid
        # dec_fea = self.W_s(st_hat).unsqueeze(1)  # bs,1,2*n_hid
        # et = et + dec_fea
        # et = torch.tanh(et)  # bs,n_seq, n_hid
        # et = self.v(et).squeeze(2)  # bs,n_seq

        # bs,seq,dim  bs,dim,1    => bs,seq,1
        enc_out = enc_out.reshape(batchsize, -1, self.hidden_size)
        et = torch.bmm(enc_out, h.reshape(batchsize, self.hidden_size, 1)).reshape(batchsize, -1)

        at = masked_softmax(vector=et,
                            mask=enc_padding_mask,
                            dim=1,
                            memory_efficient=False
                            )

        at = at.unsqueeze(1)  # bs,1,n_seq
        # Compute encoder context vector
        ct_e = torch.bmm(at, enc_out)  # bs,1,n_seq  x  bs,n_seq,n_hid =   bs, 1,  n_hid
        ct_e = ct_e.squeeze(1)  # bs,  n_hid

        # return ct_e, at, sum_temporal_srcs
        return ct_e


class DecoderStep(nn.Module):
    def __init__(self, embed_size, hidden_size, dict_comment, dropout):
        super(DecoderStep, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.W1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W = nn.Linear(self.hidden_size, dict_comment.size)
        self.wemb = nn.Embedding(dict_comment.size, self.embed_size, padding_idx=PAD)
        self.lstm = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.enc_attention = DotSimiAttn(hidden_size)
        self.dropout = dropout

    def forward(self, x_t, s_t, enc_out, enc_padding_mask):
        # print("=========== x_t.shape: ",x_t.shape )
        batchsize = x_t.shape[0]
        x = self.wemb(x_t)

        s_t = self.lstm(x, s_t)  # 因为用的 LSTMCell ,所以s_t 只是 (h,c) ,而不是nn.LSTM得到的output, (h_n, c_n)

        # tmp_h, tmp_c = s_t
        # tmp_h = tmp_h.reshape(x.size()[0], -1)
        # tmp_c = tmp_c.reshape(x.size()[0], -1)
        # s_t = (tmp_h, tmp_c)

        dec_h, dec_c = s_t
        # ct_e  = self.enc_attention(dec_h, enc_out, enc_padding_mask)
        ct_e = self.enc_attention(dec_h, enc_out, enc_padding_mask, batchsize)

        # print("dec_h.shape: ",dec_h.shape)
        # print("ct_e.shape: ",ct_e.shape)
        out = self.W(torch.tanh(F.dropout(self.W1(dec_h), self.dropout, training=self.training) +
                                F.dropout(self.W2(ct_e), self.dropout, training=self.training)))  # bs, n_vocab
        # vocab_dist = F.softmax(out, dim=1)
        logprobs = F.log_softmax(out, dim=1)

        return logprobs, s_t
