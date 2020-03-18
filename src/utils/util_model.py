import os

import networkx as nx
from src.data.tree  import get_data_tree_ptb_dgl_graph
from src.data.codesum.constants import *
from src.data.dict import Dict
# from src.model.codesum.Model import Model, ModelCritic, Seq8Tree8CFG2SeqModel, Seq8Tree8CFG2SeqModelCritic, \
#     Seq8Tree2SeqModel, Seq8CFG2SeqModel, Tree8CFG2SeqModel, Seq8Tree2SeqModelCritic
# from model.Encoder import Encoder, TreeEncoder_TreeLSTM_dgl, CFGEncoder_GGNN, CFGEncoder_GGNN_add_node_token,\
# RetrievalCodeEncoderWrapper,RetrievalCommentEncoderWrapper
from src.model.Encoder import Encoder, RetrievalCodeEncoderWrapper, RetrievalCommentEncoderWrapper
from src.module.encoder.base.encoder_ast import TreeEncoder_TreeLSTM_dgl
from src.module.encoder.encoder_cfg  import CFGEncoder_GGNN, CFGEncoder_GGNN_add_node_token
from src.module.summarization.decoder import SeqDecoder
from src.module.summarization.discriminator import Discriminator, Discriminator_LSTM

import torch.nn as nn
from src.data.codesum.codesum_dataset import *
from src.data.codeir.codeir_dataset import *
from src.data.codeir.deepcs_dataset import *
# from src.model.codeir.mman import ModelCodeRetrieval, ModelCodeRetrieval_coattn
# from src.data.codeir.codeir_dataset import CodeRetrievalDataset, codeir_collate_fn, \
#     collate_fn_code_retrieval_for_test_set, \
#     CodeRetrievalQueryDataset, codeir_collate_fn_query
# from src.data.codeir.deepcs_dataset import CodeRetrieval_DeepCS_Dataset, collate_fn_code_retrieval_deep_cs, \
#     collate_fn_code_retrieval_for_test_set_deep_cs, CodeRetrieval_DeepCS_QueryDataset, \
#     collate_fn_code_retrieval_for_query_deep_cs
from src.module.retrieval.deepcs_encoder import SeqEncoder
from src.model.retrieval.deepcs import DeepCSModel







def load_dict(opt):
    dict_code = Dict([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD], lower=opt.lower)
    # 这里添加 PAD UNK BOS EOS，但其实之前生成的dict中已经有 这四个了，所以这里初始化添加一次，后面 loadFile又添加一次，不过没啥问题，
    # 因为是添加到字典形式的变量里，重复添加的话，key和value还是不变的
    dict_comment = Dict([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD], lower=opt.lower)
    dict_code.loadFile(opt.dict_code)
    if opt.use_partion_com_voca:
        print("load partial comment voca, use_partion_com_voca: ", opt.use_partion_com_voca)
        dict_comment.loadFile(opt.dict_comment, opt.com_voca_size_to_use)
    else:
        print("load all comment voca")
        dict_comment.loadFile(opt.dict_comment)
    return_dict = [dict_code, dict_comment]

    # if opt.dataset_type == "c":
    if opt.dataset_type == "c" and opt.tree_leaf_subtoken == 0:
        dict_leaves = Dict([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD], lower=opt.lower)
        dict_leaves.loadFile(opt.ast_tree_leaves_dict)
        return_dict.append(dict_leaves)
        print("load_data in c_opt_dataset_type return_dict[2].size() :", return_dict[2].size())
        print("load_data in c_opt_dataset_type dict_leaves.size() :", dict_leaves.size())

    return return_dict


def load_data_code_sum(opt, all_dict):
    def get_max_node_num_for_one_file(cfgs):
        return max([len(list(json_dict["save_node_feature_digit"].keys())) for json_dict in cfgs])

    def get_max_node_num(train_cfgs, val_cfgs, test_cfgs):
        return max([get_max_node_num_for_one_file(cfgs) for cfgs in [train_cfgs, val_cfgs, test_cfgs]])

    dict_code, dict_comment = all_dict[0], all_dict[1]

    if opt.dataset_type == "c":
        if opt.tree_leaf_subtoken:
            dict_leaves = dict_code
            print("load_data_dict_leaves.size(): ", dict_leaves.size())
        else:
            dict_leaves = all_dict[2]
            print("load_data_c_dict_leaves.size(): ", dict_leaves.size())
    else:
        dict_leaves = dict_code
        print("load_data_dict_leaves.size(): ", dict_leaves.size())

    def _dataset_load(data_ctg):
        # data_ctg['tree_dgl'] = get_data_tree_ptb_dgl_graph(copy.deepcopy(data_ctg['tree'][:10]), dict_leaves)
        data_ctg['tree_dgl'] = get_data_tree_ptb_dgl_graph(data_ctg['tree'][:50], dict_leaves, opt.tree_leaf_subtoken)
        print("finish load dataset, size: {}".format(len(data_ctg['tree_dgl'])))
        ctg_dataset = CodeSumDataset(opt, data_ctg, dict_code, dict_comment, \
                                     n_node=n_node, n_edge_types=n_edge_types, annotation_dim=annotation_dim)
        ctg_dataloader = torch.utils.data.DataLoader(dataset=ctg_dataset, batch_size=opt.batch_size, shuffle=False,
                                                     num_workers=opt.workers, collate_fn=codesum_collate_fn)
        return ctg_dataset, ctg_dataloader,

    train_ctg = torch.load(opt.data_train_ctg)

    # n_node, n_edge_types, annotation_dim = 53, 2, 10
    # _, train_loader = _dataset_load(train_ctg)
    # train_iter = iter(train_loader)
    # batch = train_iter.__next__()

    val_ctg = torch.load(opt.data_val_ctg)
    test_ctg = torch.load(opt.data_test_ctg)

    # val_ctg_raw = copy.deepcopy(val_ctg)
    n_node = get_max_node_num(train_ctg['cfg'], val_ctg['cfg'], test_ctg['cfg'])

    if opt.dataset_type == "python":
        n_edge_types = len(list(config_about_voca.py_cfg_edge_color2index.keys()))
        annotation_dim = len(list(config_about_voca.py_cfg_node_feat2index.keys()))
    elif opt.dataset_type == "c":
        n_edge_types = len(list(config_about_voca.cfg_edge_color2index.keys()))
        annotation_dim = len(list(config_about_voca.cfg_node_color2index.keys()))

    # print(n_node, n_edge_types, annotation_dim)

    ctg_loaders = [_dataset_load(ctg) for ctg in [train_ctg, val_ctg, test_ctg]]
    datasets, data_loaders = list(zip(*ctg_loaders))
    return datasets, data_loaders,


def create_model_code_retrieval(opt, dataset, all_dict):
    def _init_param(opt, model):
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

    dict_code, dict_comment = all_dict[0], all_dict[1]
    if opt.dataset_type == "c":
        if opt.tree_leaf_subtoken:  # ast 叶子节点使用拆开后的subtoken时，vocabulary用和seq模态一样的
            dict_leaves = dict_code
        else:
            dict_leaves = all_dict[2]
    else:
        dict_leaves = dict_code

    if opt.modal_type == 'tree':
        # code_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        code_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), opt.modal_type)
        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        # model = Tree2SeqModel_dgl_CodeRetrieval(code_encoder, comment_encoder, opt)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        _init_param(opt, model)
    elif opt.modal_type == 'seq':
        # code_encoder = Encoder(opt, dict_code)
        code_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), opt.modal_type)
        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        # model = Seq2SeqModel_CodeRetrieval(code_encoder, comment_encoder, opt)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        _init_param(opt, model)
    elif opt.modal_type == "cfg":
        print(
            "create_model_CFGEncoder_GGNN \n dataset.new_annotation_dim:{} dataset.new_n_edge_types:{} dataset.new_n_node:{}". \
                format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN(opt, dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node)
        code_encoder = RetrievalCodeEncoderWrapper(opt,
                                                   CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                   dataset.new_n_edge_types, dataset.new_n_node),
                                                   opt.modal_type)
        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, comment_encoder)
        # model = CFG2SeqModel_CodeRetrieval(code_encoder, comment_encoder, opt)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)

    elif opt.modal_type == "cfg_add_node_token":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        code_encoder = RetrievalCodeEncoderWrapper(opt,
                                                   CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                  dataset.new_n_edge_types,
                                                                                  dataset.new_n_node, dict_code),
                                                   opt.modal_type)
        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, comment_encoder)
        # model = CFG2SeqModel_add_node_token_CodeRetrieval(code_encoder, comment_encoder, opt)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)

    if opt.modal_type in ['tree9coattn', 'tree9selfattn']:
        # code_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        code_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree9coattn")
        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        # model = Tree2SeqModel_dgl_CodeRetrieval(code_encoder, comment_encoder, opt)
        if opt.modal_type == "tree9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        else:
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)
        _init_param(opt, model)

    elif opt.modal_type in ["cfg_add_node_token9coattn", "cfg_add_node_token9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化

        # code_encoder = RetrievalCodeEncoderWrapper(opt,
        #                                         CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
        #                                                                           dataset.new_n_edge_types,
        #                                                                           dataset.new_n_node, dict_code),
        #                                            "cfg_add_node_token9coattn")
        if opt.use_outmlp3:
            code_encoder = RetrievalCodeEncoderWrapper(opt,
                                                       CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                      dataset.new_n_edge_types,
                                                                                      dataset.new_n_node, dict_code),
                                                       "cfg_add_node_token")
        else:
            code_encoder = RetrievalCodeEncoderWrapper(opt,
                                                       CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                      dataset.new_n_edge_types,
                                                                                      dataset.new_n_node, dict_code),
                                                       "cfg_add_node_token9coattn")

        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, comment_encoder)
        # model = CFG2SeqModel_add_node_token_CodeRetrieval(code_encoder, comment_encoder, opt)
        if opt.modal_type == "cfg_add_node_token9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        else:
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)
    ####
    elif opt.modal_type in ["cfg9coattn", "cfg9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化

        if opt.use_outmlp3:
            code_encoder = RetrievalCodeEncoderWrapper(opt,
                                                       CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                       dataset.new_n_edge_types,
                                                                       dataset.new_n_node),
                                                       "cfg")
        else:
            code_encoder = RetrievalCodeEncoderWrapper(opt,
                                                       CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                       dataset.new_n_edge_types,
                                                                       dataset.new_n_node),
                                                       "cfg9coattn")

        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, comment_encoder)
        # model = CFG2SeqModel_add_node_token_CodeRetrieval(code_encoder, comment_encoder, opt)
        if opt.modal_type == "cfg9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        else:
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)

    #####

    elif opt.modal_type in ['seq9coattn', 'seq9selfattn']:
        # code_encoder = Encoder(opt, dict_code)
        code_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), 'seq9coattn')
        # comment_encoder = Encoder(opt, dict_comment)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        # model = Seq2SeqModel_CodeRetrieval(code_encoder, comment_encoder, opt)
        if opt.modal_type == 'seq9selfattn':
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        else:
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)
        _init_param(opt, model)

    elif opt.modal_type == "seq8tree8cfg_add_node_token":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq")
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree")
        cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                  CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                 dataset.new_n_edge_types,
                                                                                 dataset.new_n_node, dict_code),
                                                  "cfg_add_node_token")
        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        # 这样初始化参数的话，虽然说 CFGEncoder_GGNN_add_node_token 内部已经有初始化参数的代码
        # 但是，包装CFGEncoder_GGNN_add_node_token 的RetrievalCodeEncoderWrapper的Linear层还需要初始化啊 TODO

        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)

    ###
    elif opt.modal_type == "seq8tree8cfg":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq")
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree")
        cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                  CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                  dataset.new_n_edge_types,
                                                                  dataset.new_n_node),
                                                  "cfg")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        # 这样初始化参数的话，虽然说 CFGEncoder_GGNN_add_node_token 内部已经有初始化参数的代码
        # 但是，包装CFGEncoder_GGNN_add_node_token 的RetrievalCodeEncoderWrapper的Linear层还需要初始化啊 TODO

        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
    ####

    elif opt.modal_type == "seq8tree":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq")
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree")
        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, tree_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)

    elif opt.modal_type == "seq8cfg_add_node_token":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq")
        cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                  CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                 dataset.new_n_edge_types,
                                                                                 dataset.new_n_node, dict_code),
                                                  "cfg_add_node_token")
        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, comment_encoder)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)

    ####
    elif opt.modal_type == "seq8cfg":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq")
        cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                  CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                  dataset.new_n_edge_types,
                                                                  dataset.new_n_node),
                                                  "cfg")
        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, comment_encoder)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
    ####

    elif opt.modal_type == "tree8cfg_add_node_token":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree")
        cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                  CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                 dataset.new_n_edge_types,
                                                                                 dataset.new_n_node, dict_code),
                                                  "cfg_add_node_token")
        code_encoder = RetrievalCodeEncoderWrapper(opt, (tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)

    ####
    elif opt.modal_type == "tree8cfg":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree")
        cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                  CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                  dataset.new_n_edge_types,
                                                                  dataset.new_n_node),
                                                  "cfg")
        code_encoder = RetrievalCodeEncoderWrapper(opt, (tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
    ####

    ####  co-attention
    elif opt.modal_type in ["seq8tree8cfg_add_node_token9coattn", "seq8tree8cfg_add_node_token9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq9coattn")
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree9coattn")

        if opt.use_outmlp3:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                     dataset.new_n_edge_types,
                                                                                     dataset.new_n_node, dict_code),
                                                      "cfg_add_node_token")
        else:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                     dataset.new_n_edge_types,
                                                                                     dataset.new_n_node, dict_code),
                                                      "cfg_add_node_token9coattn")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "seq8tree8cfg_add_node_token9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        elif opt.modal_type == "seq8tree8cfg_add_node_token9coattn":
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)

    ###
    elif opt.modal_type in ["seq8tree8cfg9coattn", "seq8tree8cfg9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq9coattn")
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree9coattn")

        # cfg_encoder = RetrievalCodeEncoderWrapper(opt,
        #                                            CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
        #                                                                           dataset.new_n_edge_types,
        #                                                                           dataset.new_n_node),
        #                                            "cfg9coattn")
        if opt.use_outmlp3:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg")
        else:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg9coattn")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "seq8tree8cfg9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        elif opt.modal_type == "seq8tree8cfg9coattn":
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)
    ####

    elif opt.modal_type in ["seq8tree9coattn", "seq8tree9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq9coattn")
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree9coattn")
        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, tree_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "seq8tree9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        elif opt.modal_type == "seq8tree9coattn":
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)

    elif opt.modal_type in ["seq8cfg_add_node_token9coattn", "seq8cfg_add_node_token9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq9coattn")

        # cfg_encoder =  RetrievalCodeEncoderWrapper(opt,
        #                                         CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
        #                                                                           dataset.new_n_edge_types,
        #                                                                           dataset.new_n_node, dict_code),
        #                                            "cfg_add_node_token9coattn")
        if opt.use_outmlp3:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                     dataset.new_n_edge_types,
                                                                                     dataset.new_n_node, dict_code),
                                                      "cfg_add_node_token")
        else:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                     dataset.new_n_edge_types,
                                                                                     dataset.new_n_node, dict_code),
                                                      "cfg_add_node_token9coattn")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "seq8cfg_add_node_token9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        elif opt.modal_type == "seq8cfg_add_node_token9coattn":
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)

    ###
    elif opt.modal_type in ["seq8cfg9coattn", "seq8cfg9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq9coattn")

        # cfg_encoder = RetrievalCodeEncoderWrapper(opt,
        #                                            CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
        #                                                                           dataset.new_n_edge_types,
        #                                                                           dataset.new_n_node),
        #                                            "cfg9coattn")
        if opt.use_outmlp3:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg")
        else:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg9coattn")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "seq8cfg9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        elif opt.modal_type == "seq8cfg9coattn":
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)
    #####

    elif opt.modal_type in ["tree8cfg_add_node_token9coattn", "tree8cfg_add_node_token9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree9coattn")

        # cfg_encoder =  RetrievalCodeEncoderWrapper(opt,
        #                                         CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
        #                                                                           dataset.new_n_edge_types,
        #                                                                           dataset.new_n_node, dict_code),
        #                                            "cfg_add_node_token9coattn")
        if opt.use_outmlp3:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                     dataset.new_n_edge_types,
                                                                                     dataset.new_n_node, dict_code),
                                                      "cfg_add_node_token")
        else:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim,
                                                                                     dataset.new_n_edge_types,
                                                                                     dataset.new_n_node, dict_code),
                                                      "cfg_add_node_token9coattn")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "tree8cfg_add_node_token9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        elif opt.modal_type == "tree8cfg_add_node_token9coattn":
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)

    ####
    elif opt.modal_type in ["tree8cfg9coattn", "tree8cfg9selfattn"]:
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        # code_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
        #                               dataset.new_n_node,dict_code)  # CFGEncoder_GGNN 已有参数初始化
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree9coattn")

        # cfg_encoder = RetrievalCodeEncoderWrapper(opt,
        #                                            CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
        #                                                                           dataset.new_n_edge_types,
        #                                                                           dataset.new_n_node),
        #                                            "cfg9coattn")
        if opt.use_outmlp3:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg")
        else:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg9coattn")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "tree8cfg9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)
        elif opt.modal_type == "tree8cfg9coattn":
            model = ModelCodeRetrieval_coattn(code_encoder, comment_encoder, opt)

    #####

    #####  Deep Code Search begin
    elif opt.modal_type == "deepcs":
        # from model.BaselineEncoder import DeepCS_BOWEncoder, DeepCS_SeqEncoder
        # from model.BaselineModel import DeepCS_Model
        model = DeepCS_Model(opt, name_encoder=DeepCS_SeqEncoder(opt, dict_code),
                             token_encoder=DeepCS_BOWEncoder(opt, dict_code),
                             desc_encoder=DeepCS_SeqEncoder(opt, dict_code))
    #####  Deep Code Search end

    ####
    if opt.model_from:
        if os.path.exists(opt.model_from):
            print("Loading from checkpoint at %s" % opt.model_from)
            checkpoint = torch.load(opt.model_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)
        else:
            print("not load pt file")

    # if opt.gpus:
    #     model.cuda()
    print("create_model_code_retrieval, opt.gpus: ", opt.gpus)
    if opt.gpus:
        model.cuda()
        print("model.cuda() ok")
        gpu_list = [int(k) for k in opt.gpus.split(",")]
        gpu_list = list(range(len(gpu_list)))
        if len(gpu_list) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_list)
            print("DataParallel ok , gpu_list: ", gpu_list)

    return model


def create_model(opt, dataset, all_dict):
    def _init_param(opt, model):
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

    dict_code, dict_comment = all_dict[0], all_dict[1]
    if opt.dataset_type == "c":
        if opt.tree_leaf_subtoken:
            print("opt.tree_leaf_subtoken: ", opt.tree_leaf_subtoken)
            dict_leaves = dict_code
            print("create_model_dict_leaves.size(): ", dict_leaves.size())
        else:
            dict_leaves = all_dict[2]
            print("create_model_c_dict_leaves.size(): ", dict_leaves.size())
    else:
        dict_leaves = dict_code
        print("create_model_dict_leaves.size(): ", dict_leaves.size())

    if opt.modal_type == 'tree':
        encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        # decoder = SeqDecoder(opt, dict_comment)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        model = Model(encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, model)
    elif opt.modal_type == 'seq':
        encoder = SeqEncoder(opt, dict_code)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        model = Model(encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, model)
    elif opt.modal_type == "cfg":
        print(
            "create_model_CFGEncoder_GGNN \n dataset.new_annotation_dim:{} dataset.new_n_edge_types:{} dataset.new_n_node:{}". \
                format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        encoder = CFGEncoder_GGNN(opt, dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node)
        # decoder = SeqDecoder(opt, dict_comment)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        if opt.init_type == "origin":
            _init_param(opt, decoder)
        model = Model(encoder, decoder, opt)
    elif opt.modal_type == "cfg_add_node_token":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
                                                 dataset.new_n_node, dict_code)  # CFGEncoder_GGNN 已有参数初始化
        # decoder = SeqDecoder(opt, dict_comment)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        if opt.init_type == "origin":
            _init_param(opt, decoder)
        model = Model(encoder, decoder, opt)
    elif opt.modal_type == 'seq8tree8cfg':
        seq_encoder = Encoder(opt, dict_code)
        tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        cfg_encoder = CFGEncoder_GGNN(opt, dataset.annotation_dim, dataset.n_edge_types,
                                      dataset.n_node)  # CFGEncoder_GGNN 已有参数初始化
        # decoder = SeqDecoder(opt, dict_comment)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        model = Seq8Tree8CFG2SeqModel(seq_encoder, tree_encoder, cfg_encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, model)

    elif opt.modal_type == 'seq8tree8cfg_add_node_token':
        seq_encoder = Encoder(opt, dict_code)
        tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        cfg_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.annotation_dim, dataset.n_edge_types, dataset.n_node,
                                                     dict_code)  # CFGEncoder_GGNN 已有参数初始化
        # decoder = SeqDecoder(opt, dict_comment)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        model = Seq8Tree8CFG2SeqModel(seq_encoder, tree_encoder, cfg_encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, model)

    elif opt.modal_type == "seq8tree":
        if opt.no_cfg:
            assert False, print("TODO!!!")
            # seq_encoder = Encoder(opt, dict_code)
            # tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
            # cfg_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.annotation_dim, dataset.n_edge_types,
            #                                              dataset.n_node, dict_code)  # CFGEncoder_GGNN 已有参数初始化
            # # decoder = SeqDecoder(opt, dict_comment)
            # if opt.has_attn:
            #     decoder = SeqDecoder_attn(opt, dict_comment)
            # else:
            #     decoder = SeqDecoder(opt, dict_comment)
            # model = Seq8Tree8CFG2SeqModel(seq_encoder, tree_encoder, cfg_encoder, decoder, opt)
            # if opt.init_type == "origin":
            #     _init_param(opt, model)

        else:
            seq_encoder = Encoder(opt, dict_code)
            tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
            # decoder = SeqDecoder(opt, dict_comment)
            if opt.has_attn:
                decoder = SeqDecoder_attn(opt, dict_comment)
            else:
                decoder = SeqDecoder(opt, dict_comment)
            model = Seq8Tree2SeqModel(seq_encoder, tree_encoder, decoder, opt)
            if opt.init_type == "origin":
                _init_param(opt, model)

    elif opt.modal_type in ['seq8cfg_add_node_token', 'seq8cfg']:
        seq_encoder = Encoder(opt, dict_code)
        cfg_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.annotation_dim, dataset.n_edge_types, dataset.n_node,
                                                     dict_code)  # CFGEncoder_GGNN 已有参数初始化
        # decoder = SeqDecoder(opt, dict_comment)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        model = Seq8CFG2SeqModel(seq_encoder, cfg_encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, model)

    elif opt.modal_type == 'tree8cfg_add_node_token':

        tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        cfg_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.annotation_dim, dataset.n_edge_types, dataset.n_node,
                                                     dict_code)  # CFGEncoder_GGNN 已有参数初始化
        # decoder = SeqDecoder(opt, dict_comment)
        if opt.has_attn:
            decoder = SeqDecoder_attn(opt, dict_comment)
        else:
            decoder = SeqDecoder(opt, dict_comment)
        model = Tree8CFG2SeqModel(tree_encoder, cfg_encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, model)

    if opt.model_from:
        if os.path.exists(opt.model_from):
            print("Loading from checkpoint at %s" % opt.model_from)

            checkpoint = torch.load(opt.model_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)
        else:
            print("not load pt file")

    if opt.gpus:
        model.cuda()

    return model


def create_disc(opt, all_dict):
    # dict_code, dict_comment = all_dict
    dict_code, dict_comment = all_dict[0], all_dict[1]
    if opt.dataset_type == "c":
        if opt.tree_leaf_subtoken:
            dict_leaves = dict_code
        else:
            dict_leaves = all_dict[2]
    else:
        dict_leaves = dict_code
    if opt.modal_type == 'tree':
        code_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        comment_encoder = Encoder(opt, dict_comment)
        disc = Discriminator(code_encoder, comment_encoder, opt)
    elif opt.modal_type == 'seq_bak':
        code_encoder = Encoder(opt, dict_code)
        comment_encoder = Encoder(opt, dict_comment)
        disc = Discriminator(code_encoder, comment_encoder, opt)
    elif opt.modal_type == 'seq-':
        # code_encoder = Encoder(opt, dict_code)
        # comment_encoder = Encoder(opt, dict_comment)
        d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]  # , 20
        d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]  # , 160
        wemb = nn.Embedding(dict_comment.size(), opt.ninp, padding_idx=PAD)
        disc = Discriminator(wemb, d_filter_sizes, d_num_filters, opt)  # , code_encoder
    elif opt.modal_type == 'seq':
        wemb = nn.Embedding(dict_comment.size(), opt.ninp, padding_idx=PAD)
        disc = Discriminator_LSTM(opt, dict_comment)  # , code_encoder
    elif opt.modal_type == 'seq8tree8cfg':
        pass
    if opt.disc_from:
        if os.path.exists(opt.disc_from):
            checkpoint = torch.load(opt.disc_from, map_location=lambda storage, loc: storage)
            disc.load_state_dict(checkpoint)
        else:
            print("not load pt file")

    if opt.gpus:
        disc.cuda()

    return disc


def create_critic(opt, dataset, all_dict):
    def _init_param(opt, critic):
        for p in critic.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

    dict_code, dict_comment = all_dict[0], all_dict[1]
    if opt.dataset_type == "c":
        if opt.tree_leaf_subtoken:
            dict_leaves = dict_code
        else:
            dict_leaves = all_dict[2]
    else:
        dict_leaves = dict_code

    if opt.modal_type == 'tree':
        encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        decoder = SeqDecoder(opt, dict_comment)
        critic = ModelCritic(encoder, decoder, opt)
        # _init_param(opt, critic)
        if opt.init_type == "origin":
            _init_param(opt, critic)
    elif opt.modal_type == 'seq':
        encoder = Encoder(opt, dict_code)
        decoder = SeqDecoder(opt, dict_comment)
        critic = ModelCritic(encoder, decoder, opt)
        # _init_param(opt, critic)
        if opt.init_type == "origin":
            _init_param(opt, critic)
    elif opt.modal_type == "cfg":
        print(
            "create_model_CFGEncoder_GGNN \n dataset.new_annotation_dim:{} dataset.new_n_edge_types:{} dataset.new_n_node:{}". \
                format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        encoder = CFGEncoder_GGNN(opt, dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node)
        decoder = SeqDecoder(opt, dict_comment)
        # _init_param(opt, decoder)
        if opt.init_type == "origin":
            _init_param(opt, decoder)
        critic = ModelCritic(encoder, decoder, opt)
    elif opt.modal_type == "cfg_add_node_token":
        print("CFG2SeqModel_add_node_token \n data.new_annotation_dim:{} data.new_n_edge_types:{} data.new_n_node:{}". \
              format(dataset.new_annotation_dim, dataset.new_n_edge_types, dataset.new_n_node))
        encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.new_annotation_dim, dataset.new_n_edge_types,
                                                 dataset.new_n_node, dict_code)  # CFGEncoder_GGNN 已有参数初始化
        decoder = SeqDecoder(opt, dict_comment)
        # _init_param(opt, decoder)
        if opt.init_type == "origin":
            _init_param(opt, decoder)
        critic = ModelCritic(encoder, decoder, opt)
    elif opt.modal_type == 'seq8tree8cfg':
        seq_encoder = Encoder(opt, dict_code)
        tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        cfg_encoder = CFGEncoder_GGNN(opt, dataset.annotation_dim, dataset.n_edge_types,
                                      dataset.n_node)  # CFGEncoder_GGNN 已有参数初始化
        decoder = SeqDecoder(opt, dict_comment)
        critic = Seq8Tree8CFG2SeqModelCritic(seq_encoder, tree_encoder, cfg_encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, critic)

    elif opt.modal_type == 'seq8tree8cfg_add_node_token':
        seq_encoder = Encoder(opt, dict_code)
        tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        cfg_encoder = CFGEncoder_GGNN_add_node_token(opt, dataset.annotation_dim, dataset.n_edge_types, dataset.n_node,
                                                     dict_code)  # CFGEncoder_GGNN 已有参数初始化
        decoder = SeqDecoder(opt, dict_comment)
        critic = Seq8Tree8CFG2SeqModelCritic(seq_encoder, tree_encoder, cfg_encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, critic)

    elif opt.modal_type == 'seq8tree':
        seq_encoder = Encoder(opt, dict_code)
        tree_encoder = TreeEncoder_TreeLSTM_dgl(opt, dict_leaves)
        decoder = SeqDecoder(opt, dict_comment)
        critic = Seq8Tree2SeqModelCritic(seq_encoder, tree_encoder, decoder, opt)
        if opt.init_type == "origin":
            _init_param(opt, critic)

    if opt.critic_from:
        if os.path.exists(opt.critic_from):
            print("Loading from checkpoint at %s" % opt.critic_from)
            checkpoint = torch.load(opt.critic_from, map_location=lambda storage, loc: storage)
            critic.load_state_dict(checkpoint)
        else:
            print("not load pt file")

    if opt.gpus:
        critic.cuda()

    return critic

## moved from main.py   end
