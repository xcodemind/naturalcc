# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.module.code2vec.base import *


class Encoder_EmbRNN(Module):

    def __init__(self, token_num: int, embed_size: int,
                 rnn_type: str, hidden_size: int, layer_num: int, dropout: float, bidirectional: bool, ) -> None:
        super(Encoder_EmbRNN, self).__init__()
        self.wemb = Encoder_Emb(token_num, embed_size, )
        self.rnn = Encoder_RNN(rnn_type, embed_size, hidden_size, layer_num, dropout, bidirectional)
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        # self.reduce_cnt = 0
        # if self.bidirectional:
        #     self.reduce_cnt+=1
        # if layer_num:
        #     self.reduce_cnt+=1
        # if self.reduce_cnt >0:
        #     self.reduce_h  = nn.Linear(self.reduce_cnt*hidden_size,hidden_size)
        #     if self.rnn_type!='GRU':
        #         self.reduce_c  = nn.Linear(self.reduce_cnt*hidden_size,hidden_size)
        self.layer_num = layer_num
        if self.bidirectional:
            self.reduce_hs = nn.ModuleList(  [nn.Linear(2*hidden_size,hidden_size)] * self.layer_num)
            if self.rnn_type!='GRU':
                self.reduce_cs  =nn.ModuleList(  [ nn.Linear(2*hidden_size,hidden_size)] * self.layer_num)
            # self.reduce_h  = nn.Linear(2*hidden_size,hidden_size)
            # if self.rnn_type!='GRU':
            #     self.reduce_c  = nn.Linear(2*hidden_size,hidden_size)
            self.reduce_out = nn.Linear(2*hidden_size,hidden_size)

        LOGGER.info("self.layer_num: {}".format(self.layer_num ))

    @classmethod
    def load_from_config(cls, config: Dict, modal: str, ) -> Any:
        instance = cls(
            token_num=config['training']['token_num'][modal],
            embed_size=config['training']['embed_size'],
            rnn_type=config['training']['rnn_type'],
            hidden_size=config['training']['rnn_hidden_size'],
            layer_num=config['training']['rnn_layer_num'],
            dropout=config['training']['dropout'],
            bidirectional=config['training']['rnn_bidirectional'],
        )
        return instance

    def init_hidden(self, batch_size: int, ) -> Any:
        return self.rnn.init_hidden(batch_size)

    def forward(self, input: torch.Tensor, input_len=None, hidden=None) -> Any:
        input_emb = self.wemb(input)
        if input_len is None:
            input_len = input.data.gt(0).sum(-1)
        output, hidden = self.rnn(input_emb, input_len, hidden)
        # LOGGER.info("hidden[0].shape: {}".format(hidden[0].shape))
        if self.bidirectional:
            if type(hidden) == tuple:

                # hidden  = (torch.tanh( self.reduce_h( F.dropout(
                #                         hidden[0].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                #                             self.dropout, training=self.training))).reshape(1,-1,self.hidden_size) ,
                #     torch.tanh( self.reduce_c(  F.dropout(
                #                         hidden[1].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                #                             self.dropout, training=self.training)) ).reshape(1,-1,self.hidden_size) )

                hidden_0 = hidden[0].view(self.layer_num, self.num_directions, -1, self.hidden_size)
                h0_list = []
                for i in range(self.layer_num):
                    h0_list.append(torch.tanh(self.reduce_hs[i](F.dropout(
                        hidden_0[i,:,:,:].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                        self.dropout, training=self.training))).reshape(1, -1, self.hidden_size))
                h0 = torch.cat(h0_list,dim=0)

                hidden_1 = hidden[1].view(self.layer_num, self.num_directions, -1, self.hidden_size)
                h1_list = []
                for i in range(self.layer_num):
                    h1_list.append(torch.tanh(self.reduce_cs[i](F.dropout(
                        hidden_1[i,:,:,:].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                        self.dropout, training=self.training))).reshape(1, -1, self.hidden_size))
                h1 = torch.cat(h1_list,dim=0)

                hidden = (h0,h1)

                # 不能只取最后一层的state，因为decoder可能层数一样，那么这里取一层，假如decoder为多层，那么要求encoder输出多层state
                # h_0 = hidden[0].view(self.layer_num, self.num_directions, -1, self.hidden_size)[-1,:,:,:]  # get the last layer's state
                # h_1 = hidden[1].view(self.layer_num, self.num_directions, -1, self.hidden_size)[-1,:,:,:]
                # hidden  = (torch.tanh( self.reduce_h( F.dropout(
                #                         h_0.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                #                             self.dropout, training=self.training))).reshape(1,-1,self.hidden_size) ,
                #     torch.tanh( self.reduce_c(  F.dropout(
                #                         h_1.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                #                             self.dropout, training=self.training)) ).reshape(1,-1,self.hidden_size) )


                # LOGGER.info("h0.shape:{} h1.shape:{}".format(h0.shape,h1.shape ))
            else:
                # hidden = torch.tanh( self.reduce_h( F.dropout(
                #     hidden[0].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                #     self.dropout, training=self.training))).reshape(1,-1,self.hidden_size)
                # h_0 = hidden[0].view(self.layer_num, self.num_directions, -1, self.hidden_size)[-1, :, :, :]
                # hidden = torch.tanh( self.reduce_h( F.dropout(
                #     h_0.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                #     self.dropout, training=self.training))).reshape(1,-1,self.hidden_size)


                hidden_0 = hidden.view(self.layer_num, self.num_directions, -1, self.hidden_size)
                h0_list = []
                for i in range(self.layer_num):
                    h0_list.append(torch.tanh(self.reduce_hs[i](F.dropout(
                        hidden_0[i,:,:,:].transpose(0, 1).contiguous().view(-1, self.hidden_size * 2),
                        self.dropout, training=self.training))).reshape(1, -1, self.hidden_size))
                hidden = torch.cat(h0_list,dim=0)

            output = torch.tanh( self.reduce_out( F.dropout( output , self.dropout, training=self.training)))

            # LOGGER.info("bidirectional, output.shape： {} ".format(output.shape ))
        # LOGGER.info("hidden[0].shape: {}".format(hidden[0].shape))
        return output, hidden


if __name__ == '__main__':
    input = torch.LongTensor([[1, 2, 4, 2], [4, 3, 0, 0]])
    encoder = Encoder_EmbRNN(token_num=10, embed_size=50,
                             rnn_type='LSTM', hidden_size=128, layer_num=3, dropout=0.1, )
    hidden = encoder.init_hidden(input.size(0))
    input, hidden = encoder(input, hidden)
    print(input.size())
