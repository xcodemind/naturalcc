# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from ncc import LOGGER
from ncc.module.code2vec.base import *
from ncc.utils.util_file import *
from ncc.data import *
from ncc.dataset.base import *
from typing import Dict, Any

class Encoder_EmbPathRNN(Module):

    def __init__(self,
                 leaf_path_k: int,
                 # for ast node, start/end ast node
                 node_token_num: int, subtoken_token_num: int, embed_size: int,
                 # for RNN, Linear, attention
                 rnn_type: str, hidden_size: int, layer_num: int, dropout: float, bidirectional: bool,
                 ) -> None:
        super(Encoder_EmbPathRNN, self).__init__()
        self.leaf_path_k = leaf_path_k
        self.node_wemb = Encoder_Emb(node_token_num, embed_size, )
        self.subtoken_wemb = Encoder_Emb(subtoken_token_num, embed_size, )
        self.rnn = Encoder_RNN(rnn_type, embed_size, hidden_size, layer_num, dropout, bidirectional)
        # 4 = bidirection + start + end
        self.fuse_linear = nn.Linear(
            embed_size * 2 + hidden_size * (2 if bidirectional else 1),
            hidden_size, bias=False)
        self.dropout = dropout

    @classmethod
    def load_from_config(cls, args: Dict, center_key='center', border_key='border', ) -> Any:
        instance = cls(
            leaf_path_k=args['dataset']['leaf_path_k'],
            node_token_num=args['training']['token_num'][center_key],
            subtoken_token_num=args['training']['token_num'][border_key],
            embed_size=args['training']['embed_size'],

            rnn_type=args['training']['rnn_type'],
            hidden_size=args['training']['rnn_hidden_size'],
            layer_num=args['training']['rnn_layer_num'],
            dropout=args['training']['dropout'],
            # bidirectional=args['training']['rnn_bidirectional'],
            bidirectional=True,
        )
        return instance

    def init_hidden(self, batch_size: int, ) -> Any:
        return self.rnn.init_hidden(batch_size)

    def forward(self,
                head: torch.Tensor, head_len: torch.Tensor, head_mask: torch.Tensor,
                center: torch.Tensor, center_len: torch.Tensor, center_mask: torch.Tensor,
                tail: torch.Tensor, tail_len: torch.Tensor, tail_mask: torch.Tensor, ) -> Any:
        # head/tail emb [bs * path_k, embed_size]
        head_emb = self.subtoken_wemb(head).sum(dim=1)
        tail_emb = self.subtoken_wemb(tail).sum(dim=1)

        center_emb = self.node_wemb(center)
        _, (center_hidden, _) = self.rnn(center_emb, center_len, is_sorted=False, )
        center_hidden = center_hidden[(-2 if self.rnn.bidirectional else -1):].transpose(dim0=0, dim1=1)
        center_hidden = center_hidden.contiguous().view(center_hidden.size(0), -1)
        fuse_emb = torch.cat([head_emb, center_hidden, tail_emb], dim=-1)
        fuse_emb = self.fuse_linear(fuse_emb)
        fuse_emb = F.dropout(fuse_emb, self.dropout, training=self.training)

        # assume its time step is path_k
        # [bs*path_k, hidden_size] => [bs, path_k, hidden_size]
        bacth_size = head.size(0) // self.leaf_path_k
        enc_output = fuse_emb.view(bacth_size, self.leaf_path_k, self.fuse_linear.out_features)
        # [bs, path_k, hidden_size] => mean => [bs, hidden_size] => expand dim => [1, bs, hidden_size]
        enc_hidden = enc_output.mean(dim=1).unsqueeze(dim=0)

        return enc_output, (enc_hidden, torch.zeros_like(enc_hidden).to(enc_hidden.device)), \
               torch.ones(bacth_size, enc_output.size(1)).to(enc_hidden.device)


if __name__ == '__main__':
    import socket

    run_dir = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/run/summarization/unilang/code2seq'
    yaml_file = os.path.join(run_dir, 'python.yml')
    args = load_args(yaml_file)
    CUR_SCRIPT = sys.argv[0].split('/')[-1].split('.')[0]
    LOGGER.info("Start {}.py ... =>  PID: {}".format(CUR_SCRIPT, os.getpid()))
    LOGGER.info('Load arguments in {}'.format(yaml_file))

    token_dicts = TokenDicts(args['dicts'])
    args['training']['token_num'] = token_dicts.size

    file_dir = args['dataset']['dataset_dir']
    data_lng = args['dataset']['source']['dataset_lng'][0]
    code_modalities = args['training']['code_modalities']
    mode = args['dataset']['source']['mode'][0]
    token_dicts = TokenDicts(args['dicts'])

    dataset = sBaseDataset(file_dir, data_lng, code_modalities, mode, token_dicts,
                           None, None, None, False)

    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=args['training']['batch_size'], shuffle=False,
                             collate_fn=sbase_collate_fn, )
    data_iter = iter(data_loader)
    batch_data = data_iter.__next__()

    encoder = Encoder_EmbPathRNN.load_from_config(args)
    LOGGER.info(encoder)
    enc_output, hidden, _ = encoder(*batch_data['path'])
    print(enc_output.size())
    print(hidden[0].size())
