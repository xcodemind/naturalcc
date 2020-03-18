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
    args = get_args()
    # sudo nvidia-smi -pm 1

    # for debug
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    # nohup python -u ./run/retrieval/unilang/nbow/main.py > ./run/retrieval/unilang/nbow/retrieval_*.log 2>&1 &
    # nohup python -u ./run/retrieval/unilang/nbow/main.py > ./run/retrieval/unilang/nbow/retrieval300_*.log 2>&1 &
    # args = Argues('config.yml', 'retrieval', 'unilang', 'nbow', 'train_sl', 'source', True)  # train_sl
    # args = Argues('go.yml', 'retrieval', 'unilang', 'nbow', 'test', 'source', True)  # test
    LOGGER.info(args)

    config, dataset, = load_config_dataset(args, XlangDataloader, rBaseDataset, rbase_collate_fn)

    # unilang-language
    src_lng = config['dataset'][args.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    model = build_model(config, NBOW(config))
    LOGGER.info(model)

    # train_sl
    if args.train_mode == 'all' or args.train_mode == 'train_sl':
        re_criterion = RetrievalNLLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim']) \
            (model.parameters(), config['sl']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(config['dataset'][args.dataset_type]['dataset_lng']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        sl_trainer = SLTrainer(config)
        best_model = sl_trainer.train(model, unilang_dataset, re_criterion, optimizer, SAVE_DIR=save_dir, )

    # test
    if args.train_mode == 'all' or args.train_mode == 'test':
        for metric, model_path in best_model.items():
            for pool_size in [-1, 1000]:
                model.load_state_dict(torch.load(best_model[metric], map_location=lambda storage, loc: storage))
                LOGGER.info('test on {}, model weights of best {} from {}'.format(unilang_dataset, metric, model_path))
                _, mmr, _, ndcg, pool_size = Evaluator.retrieval_eval(model, unilang_dataset['test'],
                                                                      pool_size=pool_size)
                headers = ['MRR@{}'.format(pool_size), 'NDCG@{}'.format(pool_size), ]
                result_table = [[round(i, 4) for i in [mmr, ndcg]]]
                LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                      tablefmt=model.config['common'][
                                                                          'result_table_format'])))


if __name__ == '__main__':
    main()
