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
    # nohup python -u ./run/retrieval/unilang/deepcs/main.py > ./run/retrieval/unilang/deepcs/retrieval_*.log 2>&1 &
    Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    # args = Argues('ruby.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args = Argues('php.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args = Argues('python.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args = Argues('java.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args = Argues('javascript.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl
    # args = Argues('go.yml', 'retrieval', 'unilang', 'deepcs', 'train_sl', 'source', True)  # train_sl

    args = Argues('javascript.yml', 'retrieval', 'unilang', 'deepcs', 'test', 'source', True)  # train_sl
    LOGGER.info(args)

    config, dataset, = load_config_dataset(args, XlangDataloader, rBaseDataset, rbase_collate_fn)

    # unilang-language
    src_lng = config['dataset'][args.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    model = build_model(config, DeepCodeSearch(config))
    LOGGER.info(model)

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
