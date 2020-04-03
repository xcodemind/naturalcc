# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from ncc.metric.base import *
from ncc.model.retrieval.unilang import *
from ncc.trainer.retrieval.unilang import *
from ncc.eval import *

from ncc.eval.evaluator import *
from collections import namedtuple
import os
import torch

def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    # args = get_args()

    # for debug
    Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type')
    # nohup python -u ./run/retrieval/unilang/ahn/main.py > ./run/retrieval/unilang/ahn/retrieval_sl.log 2>&1 &
    # args = Argues('ruby.yml', 'retrieval', 'unilang', 'ahn', 'train_sl', 'source')  # train_sl
    # nohup python -u ./run/retrieval/unilang/ahn/main.py > ./run/retrieval/unilang/ahn/retrieval_al.log 2>&1 &
    # args = Argues('ruby.yml', 'retrieval', 'unilang', 'ahn', 'train_al', 'source')  # train_al
    args_ = Argues('ruby.yml', 'retrieval', 'unilang', 'ahn', 'train_hash', 'source')  # train_al
    # args = Argues('ruby.yml', 'retrieval', 'unilang', 'ahn', 'test', 'source')  # train_sl
    LOGGER.info(args_)

    args, dataset, = load_args_dataset(args_, XlangDataloader, rBaseDataset, rbase_collate_fn)

    # unilang-language
    src_lng = args['dataset'][args_.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args_.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    TRAIN_NUM = unilang_dataset.size['train']
    model = build_model(args, AHN_NBOW(args, TRAIN_NUM))

    if args_.train_mode == 'train_sl':
        # to evaluate our model structure correct
        re_criterion = RetrievalNLLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim']) \
            (model.trainable_parameters(), args['sl']['lr'])

        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'sl_{}'.format('-'.join(args['dataset'][args_.dataset_type]['dataset_lng'])),
                                args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        sl_trainer = SLTrainer(args)
        sl_trainer.train(model, unilang_dataset, re_criterion, optimizer, SAVE_DIR=None, )

    elif args_.train_mode == 'train_al':
        re_criterion = RetrievalNLLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim']) \
            (model.trainable_parameters(), args['sl']['lr'])
        disc_optimizer = getattr(torch.optim, args['al']['optim']) \
            (model.disc_parameters(), args['al']['lr'])
        torch.optim.Adam
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'al_{}'.format('-'.join(args['dataset'][args_.dataset_type]['dataset_lng'])),
                                args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        ah_trainer = AHTrainer(args)
        ah_trainer.train_al(model, unilang_dataset, re_criterion, optimizer, disc_optimizer,
                            SAVE_DIR=None, )


    elif args_.train_mode == 'train_hash':
        code_optimizer = getattr(torch.optim, args['hash']['optim']) \
            (model.code_parameters(), args['hash']['lr'])
        cmnt_optimizer = getattr(torch.optim, args['hash']['optim']) \
            (model.cmnt_parameters(), args['hash']['lr'])

        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(args['dataset'][args_.dataset_type]['dataset_lng']), args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        ah_trainer = AHTrainer(args)
        ah_trainer.train_hash(model, unilang_dataset, code_optimizer, cmnt_optimizer, SAVE_DIR=None, )

    elif args_.train_mode == 'train_ah':
        disc_optimizer = getattr(torch.optim, args['al']['optim']) \
            (model.disc_parameters(), args['al']['lr'])
        code_optimizer = getattr(torch.optim, args['hash']['optim']) \
            (model.code_parameters(), args['hash']['lr'])
        cmnt_optimizer = getattr(torch.optim, args['hash']['optim']) \
            (model.cmnt_parameters(), args['hash']['lr'])

        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(args['dataset'][args_.dataset_type]['dataset_lng']), args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        ah_trainer = AHTrainer(args)
        ah_trainer.train(model, unilang_dataset,
                         disc_optimizer, code_optimizer, cmnt_optimizer,
                         SAVE_DIR=save_dir, )


    elif args_.train_mode == 'test':
        acc, mmr, map, ndcg, pool_size = Evaluator.retrieval_eval(model, unilang_dataset['test'])
        headers = ['ACC@{}'.format(pool_size), 'MRR@{}'.format(pool_size), 'MAP@{}'.format(pool_size),
                   'NDCG@{}'.format(pool_size), ]
        result_table = [[round(i, 4) for i in [acc, mmr, map, ndcg]]]
        LOGGER.info('Evaluation results:\n{}'.format(
            tabulate(result_table, headers=headers, tablefmt=model.args['common']['result_table_format']))
        )
    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
