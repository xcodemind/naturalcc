# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.module.code2vec.base import Encoder_Emb

import dgl


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int) -> None:
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges: Any) -> Dict:
        # aggregate neighbors’ representations
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes: Any) -> Dict:
        # tranform aggregated representations from neighbors
        # LOGGER.debug(nodes.mailbox['h'].size())
        h_cat = nodes.mailbox['h'].reshape(nodes.mailbox['h'].size(0), -1)
        f = torch.sigmoid(self.U_f(h_cat)).reshape(*nodes.mailbox['h'].size())
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return_iou = self.U_iou(h_cat)
        return {'iou': return_iou, 'c': c}

    def apply_node_func(self, nodes: Any) -> Dict:
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        # LOGGER.debug(h.size())
        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int) -> None:
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges: Any) -> Dict:
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes: Any) -> Dict:
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes: Any) -> Dict:
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class Encoder_EmbTreeRNN(nn.Module):
    def __init__(self, token_num: int, embed_size: int,
                 cell_type: str, hidden_size: int, tree_leaf_subtoken: bool, ) -> None:
        super(Encoder_EmbTreeRNN, self).__init__()
        # embedding params
        self.token_num = token_num
        self.embed_size = embed_size
        # rnn params
        cell = TreeLSTMCell if cell_type == 'nary' else ChildSumTreeLSTMCell
        self.hidden_size = hidden_size
        self.tree_leaf_subtoken = tree_leaf_subtoken

        self.wemb = Encoder_Emb(self.token_num, self.embed_size)
        self.cell = cell(self.embed_size, self.hidden_size)

    @classmethod
    def load_from_config(cls, config: Dict, modal: str, ) -> Any:
        instance = cls(
            token_num=config['training']['token_num'][modal],
            embed_size=config['training']['embed_size'],
            cell_type=config['training']['tree_lstm_cell_type'],
            hidden_size=config['training']['rnn_hidden_size'],
            tree_leaf_subtoken=config['dataset']['tree_leaf_subtoken'],
        )
        return instance

    def forward(self, batch, enc_hidden, list_root_index, list_num_node) -> Any:

        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # print("TreeEncoder_TreeLSTM_dgl_forward")
        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)

        if self.tree_leaf_subtoken:
            emb_subtoken = self.wemb(batch.wordid * batch.mask.reshape(*batch.mask.shape, 1))
            wemb = torch.sum(emb_subtoken, dim=1)  # feed embedding
        else:
            wemb = self.wemb(batch.wordid * batch.mask)  # feed embedding
        g.ndata['iou'] = self.cell.W_iou(wemb) * batch.mask.float().unsqueeze(-1)
        # LOGGER.debug(g.ndata['iou'].size())
        g.ndata['h'], g.ndata['c'], = enc_hidden

        dgl.prop_nodes_topo(g)

        all_node_h_in_batch = g.ndata.pop('h')
        all_node_c_in_batch = g.ndata.pop('c')

        batch_size = list_root_index.size()[0]
        root_node_h_in_batch, root_node_c_in_batch = [], []
        add_up_num_node = 0
        for _i in range(len(list_root_index)):
            if _i - 1 < 0:
                add_up_num_node = 0
            else:
                add_up_num_node += list_num_node[_i - 1]
            idx_to_query = list_root_index[_i] + add_up_num_node
            root_node_h_in_batch.append(all_node_h_in_batch[idx_to_query])
            root_node_c_in_batch.append(all_node_c_in_batch[idx_to_query])

        # root_node_h_in_batch = torch.cat(root_node_h_in_batch).reshape(1, len(root_node_h_in_batch), -1)
        # root_node_c_in_batch = torch.cat(root_node_c_in_batch).reshape(1, len(root_node_c_in_batch), -1)
        root_node_h_in_batch = torch.cat(root_node_h_in_batch).reshape(batch_size, -1)
        root_node_c_in_batch = torch.cat(root_node_c_in_batch).reshape(batch_size, -1)

        tree_output = torch.zeros(batch_size, max(list_num_node), root_node_h_in_batch.shape[-1]).cuda()
        add_up_node_num = 0
        for _i in range(batch_size):
            node_num = list_num_node[_i]
            this_sample_h = all_node_h_in_batch[add_up_node_num:add_up_node_num + list_num_node[_i]]. \
                reshape(node_num, -1)
            add_up_node_num += list_num_node[_i]
            tree_output[_i, :node_num, :] = this_sample_h

        return tree_output, (root_node_h_in_batch, root_node_c_in_batch)

    def init_hidden(self, batch_size: int) -> Tuple:
        weight = next(self.parameters()).data
        return (
            weight.new(batch_size, self.hidden_size).zero_().requires_grad_(),
            weight.new(batch_size, self.hidden_size).zero_().requires_grad_()
        )
