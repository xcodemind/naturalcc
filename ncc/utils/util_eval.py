# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

import copy
import numpy as np

from eval.summarization.bleu.bleu import Bleu
from eval.summarization.cider.cider import Cider
from eval.summarization.meteor.meteor import Meteor
from eval.summarization.rouge.rouge import Rouge

from ncc.utils.utils import *

from ncc.data.token_dicts import TokenDicts
import ujson

# from ncc.log.log import get_logger
#
# LOGGER = get_logger()
from ncc import LOGGER

def dump_preds(src_comments: List, trg_comments: List, pred_comments: List,
               src_code_all: List, pred_filename: str, ) -> None:
    # write
    with open(pred_filename, 'w', encoding='utf-8') as f:
        for i, (src_cmt, pred_cmt, trg_cmt, src_code,) in \
                enumerate(zip(src_comments, pred_comments, trg_comments, src_code_all, )):
            f.write('=============================> {}\n'.format(i))
            # code
            f.write('[src code]\n{}\n'.format(src_code))
            # f.write('[trg code]\n{}\n'.format(' '.join(trg_code)))
            # comment
            f.write('[src cmnt]\n{}\n'.format(src_cmt))
            f.write('[pre cmnt]\n{}\n'.format(' '.join(pred_cmt)))
            f.write('[trg cmnt]\n{}\n'.format(' '.join(trg_cmt)))
            f.write('\n\n')
    LOGGER.info("Write source/predict/target comments into {}, size: {}".format(pred_filename, len(src_comments)))


def eval_metrics(src_comments: List, trg_comments: List, pred_comments: List,
                 src_code_all: List, oov_vocab: List,
                 token_dicts: TokenDicts, pred_filename=None, metrics=METRICS, ) -> Tuple:
    preds, trgs = {}, {}
    srcs_return, trgs_return, preds_return = {}, {}, {}
    new_pred_comments = [None] * len(pred_comments)

    for i, (src, trg, pred,) in enumerate(zip(src_comments, trg_comments, pred_comments, )):
        pred = clean_up_sentence(pred, remove_EOS=True)
        if oov_vocab is not None:
            pred = indices_to_words(pred, token_dicts['comment'], oov_vocab[i])
        else:
            pred = token_dicts['comment'].to_labels(pred, EOS_WORD)
        new_pred_comments[i] = pred

        preds_return[i] = copy.deepcopy(pred)
        trgs_return[i] = copy.deepcopy(trg)
        srcs_return[i] = src

        preds[i] = [' '.join(pred)]
        trgs[i] = [' '.join(trg)]

    # eval score
    if 'bleu' in metrics:
        _, bleu = Bleu(4).compute_score(trgs, preds)
        bleu1, bleu2, bleu3, bleu4, = bleu
        # print('bleu1-: ', bleu1)
    else:
        bleu1, bleu2, bleu3, bleu4 = \
            [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments)
    if 'meteor' in metrics:
        _, meteor = Meteor().compute_score(trgs, preds)
    else:
        meteor = [0.0] * len(src_comments)

######## TODO  debug
    # filepath = '/data/wanyao/work/p/nlp/naturalcodev2/rouge_pred.txt'
    # LOGGER.info("for debug!!!!!! , filepath: {}".format(filepath)  )
    # with open(filepath, 'w', encoding='utf-8') as f:
    #     for i in range(len(preds)):
    #         f.write('{}'.format( preds[i][0]+'\n' if preds[i][0][-1] != '\n' else preds[i][0]))
    #
    # filepath = '/data/wanyao/work/p/nlp/naturalcodev2/rouge_tgt.txt'
    # LOGGER.info("for debug!!!!!! , filepath: {}".format(filepath)  )
    # with open(filepath, 'w', encoding='utf-8') as f:
    #     for i in range(len(trgs)):
    #         f.write('{}'.format( trgs[i][0]+'\n' if trgs[i][0][-1] != '\n' else trgs[i][0]))
    #
    # LOGGER.info("begin_debug_rouge")
    # for i in range(0,len(preds)-1,2):
    #     trgstmp={i:trgs[i],i+1:trgs[i+1]}
    #     predsstmp={i:preds[i],i+1:preds[i+1]}
    #     if i+2 == len(preds)-1 :
    #         trgstmp.update({i+2: trgs[i+2]})
    #         predsstmp.update({i+2: preds[i+2]})
    #     LOGGER.info("debug_rouge {}/{} begin  i:{}\ntrgstmp:\n{}\npredsstmp:\n{}\n".format(i, len(preds),i,trgstmp ,predsstmp))
    #     rouge, _ = Rouge().compute_score(trgstmp, predsstmp )  #
    #     rouge1, rouge2, rouge3, rouge4, rougel, _, _, _ = [[i] for i in rouge]
    #     LOGGER.info("debug_rouge {}/{} ok".format(i, len(preds)))
    # LOGGER.info("finish_debug_rouge")
######### 0 1 2 3 4 5 6

    if 'rouge' in metrics:
        # print('rouge-trgs: ', trgs)
        # print('rouge-preds: ', preds)
        rouge, _ = Rouge().compute_score(trgs, preds)  #
        # assert False
        # print('rouge: ', rouge)
        # print('_: ', _)
        rouge1, rouge2, rouge3, rouge4, rougel, _, _, _ = [[i] for i in rouge]
        # print('rouge1-: ', rouge1)
    else:
        rouge1, rouge2, rouge3, rouge4, rougel = [0.0] * len(src_comments), [0.0] * len(src_comments), \
                                                 [0.0] * len(src_comments), [0.0] * len(src_comments), \
                                                 [0.0] * len(src_comments),
    if 'cider' in metrics:
        _, cider = Cider().compute_score(trgs, preds)
    else:
        cider = [0.0] * len(src_comments)

    if pred_filename is not None:
        dump_preds(src_comments, trg_comments, new_pred_comments, src_code_all, pred_filename, )
    else:
        pass

    return bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, srcs_return, trgs_return, preds_return,


def dump_all(src_comments, trg_comments, new_pred_comments, src_code_all,
             tok_len, comment_len, ast_len,
             bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider,
             pred_filename, ):
    with open(pred_filename, 'w') as writer:
        for ind in range(len(src_comments)):
            code_snippet = ujson.dumps({
                'src_comment': src_comments[ind],
                'trg_comment': trg_comments[ind],
                'pred_comment': new_pred_comments[ind],
                'src_code': src_code_all[ind],

                'tok_len': tok_len[ind],
                'comment_len': comment_len[ind],
                'ast_len': ast_len[ind],

                'bleu1': bleu1[ind],
                'bleu2': bleu2[ind],
                'bleu3': bleu3[ind],
                'bleu4': bleu4[ind],
                'meteor': meteor[ind],
                'rouge1': rouge1[ind],
                'rouge2': rouge2[ind],
                'rouge3': rouge3[ind],
                'rouge4': rouge4[ind],
                'rougel': rougel[ind],
                'cider': cider[ind],
            })
            writer.write(code_snippet + '\n')


def eval_per_metrics(src_comments: List, trg_comments: List, pred_comments: List,
                     src_code_all: List, oov_vocab: List, tok_len: List, comment_len: List, ast_len: List,
                     token_dicts: TokenDicts, pred_filename=None, metrics=METRICS, ):
    # because calculate rouge one by one is too slow, so we only consider cider
    assert pred_filename is not None

    preds, trgs = {}, {}
    srcs_return, trgs_return, preds_return = {}, {}, {}
    # rouge1, rouge2, rouge3, rouge4, rougel = [], [], [], [], []
    new_pred_comments = [None] * len(pred_comments)

    for i, (src, trg, pred,) in enumerate(zip(src_comments, trg_comments, pred_comments, )):
        pred = clean_up_sentence(pred, remove_EOS=True)
        if oov_vocab is not None:
            pred = indices_to_words(pred, token_dicts['comment'], oov_vocab[i])
        else:
            pred = token_dicts['comment'].to_labels(pred, EOS_WORD)
        new_pred_comments[i] = pred

        preds_return[i] = copy.deepcopy(pred)
        trgs_return[i] = copy.deepcopy(trg)
        srcs_return[i] = src

        preds[i] = [' '.join(pred)]
        trgs[i] = [' '.join(trg)]

    # eval score
    if 'bleu' in metrics:
        _, bleu = Bleu(4).compute_score(trgs, preds)
        bleu1, bleu2, bleu3, bleu4, = bleu
        # print('bleu1-: ', bleu1)
    else:
        bleu1, bleu2, bleu3, bleu4 = \
            [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments)
    if 'meteor' in metrics:
        _, meteor = Meteor().compute_score(trgs, preds)
    else:
        meteor = [0.0] * len(src_comments)
    if 'rouge' in metrics:
        # print('rouge-trgs: ', trgs)
        # print('rouge-preds: ', preds)
        rouge1, rouge2, rouge3, rouge4, rougel = [], [], [], [], []
        for ind in range(len(trgs)):
            rouge, _ = Rouge().compute_score({ind: trgs[ind]}, {ind: preds[ind]}, )  #
            # assert False
            # print('rouge: ', rouge)
            # print('_: ', _)
            _rouge1, _rouge2, _rouge3, _rouge4, _rougel, _, _, _ = [[i] for i in rouge]
            # print('rouge1-: ', rouge1)
            rouge1.extend(_rouge1)
            rouge2.extend(_rouge2)
            rouge3.extend(_rouge3)
            rouge4.extend(_rouge4)
            rougel.extend(_rougel)
    else:
        rouge1, rouge2, rouge3, rouge4, rougel = [0.0] * len(src_comments), [0.0] * len(src_comments), \
                                                 [0.0] * len(src_comments), [0.0] * len(src_comments), \
                                                 [0.0] * len(src_comments),
    if 'cider' in metrics:
        _, cider = Cider().compute_score(trgs, preds)
    else:
        cider = [0.0] * len(src_comments)

    dump_all(src_comments, trg_comments, new_pred_comments, src_code_all,
             tok_len, comment_len, ast_len,
             bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider,
             pred_filename, )


def calculate_scores_multi_dataset(num_dataset, dataset_id_all, token_dicts, comment_pred_all, comment_target_all,
                                   oov_voca, metrics):
    # comment_target_all:  [[],[]]  every list is raw comment
    res, gts, oov = {}, {}, {}
    # res_all, gts_all = {}, {}
    res_all, gts_all = [], []
    metric = {}
    metric['all'] = {}
    for n in range(num_dataset):
        # res[n] = {}
        # gts[n] = {}
        res[n] = []
        gts[n] = []
        oov[n] = []
        metric[n] = {}
    for i in range(len(comment_pred_all)):
        pred, target = comment_pred_all[i], comment_target_all[i]
        oov_list = oov_voca[i]
        # pred = clean_up_sentence(pred, remove_unk=False, remove_eos=True)
        # pred = indices_to_words(pred, token_dicts['comment'], oov_voca[i])
        res[dataset_id_all[i]].append(pred)
        gts[dataset_id_all[i]].append(target)
        oov[dataset_id_all[i]].append(oov_list)
        res_all.append(pred)
        gts_all.append(target)
        # res_all[i] = [' '.join(pred)]
        # gts_all[i] = [' '.join(target)]
    # LOGGER.info("in calculate_scores_multi_dataset before 1st eval_metrics , len(oov_voca): {}".format(len(oov_voca)))
    bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, \
    cider, srcs_return, trgs_return, preds_return = \
        eval_metrics(src_comments=['n'] * len(gts_all), trg_comments=gts_all, pred_comments=res_all, src_code_all=[],
                     oov_vocab=oov_voca, token_dicts=token_dicts, pred_filename=None, metrics=metrics)
    bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
        map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
                                                    rouge1, rouge2, rouge3, rouge4, rougel, cider,))
    metric['all']['bleu1'], metric['all']['bleu2'], metric['all']['bleu3'], \
    metric['all']['bleu4'], metric['all']['meteor'], \
    metric['all']['rouge1'], metric['all']['rouge2'], metric['all']['rouge3'], metric['all']['rouge4'], \
    metric['all']['rougel'], \
    metric['all']['cider'] = bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider
    ##

    # LOGGER.info("in calculate_scores_multi_dataset before 2nd eval_metrics , len(oov_voca): {}".format(len(oov_voca)))

    for n in range(num_dataset):
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, \
        cider, srcs_return, trgs_return, preds_return = \
            eval_metrics(src_comments=['n'] * len(gts[n]), trg_comments=gts[n], pred_comments=res[n],
                         src_code_all=[], oov_vocab=oov[n], token_dicts=token_dicts, pred_filename=None,
                         metrics=metrics)
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
            map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
                                                        rouge1, rouge2, rouge3, rouge4, rougel, cider,))

        metric[n]['bleu1'] = bleu1

    return metric


################################################################
# retrieval eval func

# ref:
# https://github.com/guxd/deep-code-search/blob/a00206ed48543d2f543bd26b8a80f32a0404917e/pytorch/train.py#L178-L212
################################################################

import math
import torch


def normalize(data: torch.Tensor, ) -> torch.Tensor:
    """normalize matrix by rows"""
    return data / data.norm(dim=-1, keepdim=True)


def ACC(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1
    return sum / float(len(real))


def MAP(real, predict):
    sum = 0.0
    for id, val in enumerate(real):
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + (id + 1) / float(index + 1)
    return sum / float(len(real))


def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg

################################################################
