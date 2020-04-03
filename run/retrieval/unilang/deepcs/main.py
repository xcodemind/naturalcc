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
    # nohup python -u ./run/retrieval/unilang/deepcs/main.py > ./run/retrieval/unilang/deepcs/retrieval_*.log 2>&1 &
    Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    # args_ = Argues('ruby.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args_ = Argues('php.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args_ = Argues('python.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args_ = Argues('java.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args_ = Argues('javascript.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args_ = Argues('go.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl

    args_ = Argues('javascript.yml', 'retrieval', 'unilang', 'deepcs', 'test', 'source', True)  # train_sl
    LOGGER.info(args_)

    args, dataset, = load_args_dataset(args_, XlangDataloader, rBaseDataset, rbase_collate_fn)

    # unilang-language
    src_lng = args['dataset'][args_.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args_.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    model = build_model(args, DeepCodeSearch(args))
    LOGGER.info(model)

    if args_.train_mode == 'train_sl':
        re_criterion = RetrievalNLLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim']) \
            (model.parameters(), args['sl']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(args['dataset'][args_.dataset_type]['dataset_lng']), args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        sl_trainer = SLTrainer(args)
        sl_trainer.train(model, unilang_dataset, re_criterion, optimizer, SAVE_DIR=save_dir, )


    elif args_.train_mode == 'test':
        acc, mmr, map, ndcg, pool_size = Evaluator.retrieval_eval(model, unilang_dataset['test'])
        headers = ['ACC@{}'.format(pool_size), 'MRR@{}'.format(pool_size), 'MAP@{}'.format(pool_size),
                   'NDCG@{}'.format(pool_size), ]
        result_table = [[round(i, 4) for i in [acc, mmr, map, ndcg]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                              tablefmt=model.args['common'][
                                                                  'result_table_format'])))
    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
