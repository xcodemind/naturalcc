# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from ncc.metric.base import *
from ncc.model.retrieval.unilang import *
from ncc.trainer.retrieval.unilang import *
from ncc.eval import *

from ncc.eval.evaluator import *


def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    # args = get_args()

    # for debug
    # nohup python -u ./run/retrieval/unilang/mman/main.py > ./run/retrieval/unilang/mman/retrieval_*.log 2>&1 &
    Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    args = Argues('ruby-tok8path.yml', 'retrieval', 'unilang', 'mman', 'train_sl', 'source', True)  # train_sl
    # args = Argues('ruby-tok8path.yml', 'retrieval', 'unilang', 'mman', 'test', 'source', True)  # train_sl
    LOGGER.info(args)

    config, dataset, = load_config_dataset(args, XlangDataloader, rBaseDataset, rbase_collate_fn)

    # unilang-language
    src_lng = config['dataset'][args.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    model = build_model(config, MMAN(config))
    LOGGER.info(model)
    '''
    tok feature: torch.Size([128, 571, 512])
    tok hidden: torch.Size([1, 128, 512]) torch.Size([1, 128, 512])
    tok feature after attn: torch.Size([128, 512])
    ast feature: torch.Size([128, 143, 512])
    ast hidden: torch.Size([128, 512]) torch.Size([128, 512])
    ast feature after attn: torch.Size([128, 512])
    concate torch.Size([128, 1024])
    fuse: torch.Size([128, 512])
    '''

    if args.train_mode == 'train_sl':
        re_criterion = RetrievalNLLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim']) \
            (model.parameters(), config['sl']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(config['dataset'][args.dataset_type]['dataset_lng']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        sl_trainer = SLTrainer(config)
        sl_trainer.train(model, unilang_dataset, re_criterion, optimizer, SAVE_DIR=save_dir, )


    elif args.train_mode == 'test':
        acc, mmr, map, ndcg, pool_size = Evaluator.retrieval_eval(model, unilang_dataset['test'])
        headers = ['ACC@{}'.format(pool_size), 'MRR@{}'.format(pool_size), 'MAP@{}'.format(pool_size),
                   'NDCG@{}'.format(pool_size), ]
        result_table = [[round(i, 4) for i in [acc, mmr, map, ndcg]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                              tablefmt=model.config['common'][
                                                                  'result_table_format'])))
    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
