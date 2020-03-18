# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from src.metric.base import *
from src.model.retrieval.unilang import *
from src.trainer.retrieval.unilang import *
from src.eval import *

from src.eval.evaluator import *


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
    args = Argues('ruby.yml', 'retrieval', 'unilang', 'ahn', 'train_hash', 'source')  # train_al
    # args = Argues('ruby.yml', 'retrieval', 'unilang', 'ahn', 'test', 'source')  # train_sl
    LOGGER.info(args)

    config, dataset, = load_config_dataset(args, XlangDataloader, rBaseDataset, rbase_collate_fn)

    # unilang-language
    src_lng = config['dataset'][args.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    TRAIN_NUM = unilang_dataset.size['train']
    model = build_model(config, AHN_NBOW(config, TRAIN_NUM))

    if args.train_mode == 'train_sl':
        # to evaluate our model structure correct
        re_criterion = RetrievalNLLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim']) \
            (model.trainable_parameters(), config['sl']['lr'])

        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'sl_{}'.format('-'.join(config['dataset'][args.dataset_type]['dataset_lng'])),
                                args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        sl_trainer = SLTrainer(config)
        sl_trainer.train(model, unilang_dataset, re_criterion, optimizer, SAVE_DIR=None, )

    elif args.train_mode == 'train_al':
        re_criterion = RetrievalNLLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim']) \
            (model.trainable_parameters(), config['sl']['lr'])
        disc_optimizer = getattr(torch.optim, config['al']['optim']) \
            (model.disc_parameters(), config['al']['lr'])
        torch.optim.Adam
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'al_{}'.format('-'.join(config['dataset'][args.dataset_type]['dataset_lng'])),
                                args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        ah_trainer = AHTrainer(config)
        ah_trainer.train_al(model, unilang_dataset, re_criterion, optimizer, disc_optimizer,
                            SAVE_DIR=None, )


    elif args.train_mode == 'train_hash':
        code_optimizer = getattr(torch.optim, config['hash']['optim']) \
            (model.code_parameters(), config['hash']['lr'])
        cmnt_optimizer = getattr(torch.optim, config['hash']['optim']) \
            (model.cmnt_parameters(), config['hash']['lr'])

        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(config['dataset'][args.dataset_type]['dataset_lng']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        ah_trainer = AHTrainer(config)
        ah_trainer.train_hash(model, unilang_dataset, code_optimizer, cmnt_optimizer, SAVE_DIR=None, )

    elif args.train_mode == 'train_ah':
        disc_optimizer = getattr(torch.optim, config['al']['optim']) \
            (model.disc_parameters(), config['al']['lr'])
        code_optimizer = getattr(torch.optim, config['hash']['optim']) \
            (model.code_parameters(), config['hash']['lr'])
        cmnt_optimizer = getattr(torch.optim, config['hash']['optim']) \
            (model.cmnt_parameters(), config['hash']['lr'])

        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(config['dataset'][args.dataset_type]['dataset_lng']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        ah_trainer = AHTrainer(config)
        ah_trainer.train(model, unilang_dataset,
                         disc_optimizer, code_optimizer, cmnt_optimizer,
                         SAVE_DIR=save_dir, )


    elif args.train_mode == 'test':
        acc, mmr, map, ndcg, pool_size = Evaluator.retrieval_eval(model, unilang_dataset['test'])
        headers = ['ACC@{}'.format(pool_size), 'MRR@{}'.format(pool_size), 'MAP@{}'.format(pool_size),
                   'NDCG@{}'.format(pool_size), ]
        result_table = [[round(i, 4) for i in [acc, mmr, map, ndcg]]]
        LOGGER.info('Evaluation results:\n{}'.format(
            tabulate(result_table, headers=headers, tablefmt=model.config['common']['result_table_format']))
        )
    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
