# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from src.metric.base import *
from src.model.summarization.unilang import *
from src.trainer.summarization.unilang import *
from src.trainer.summarization.xlang import *
from src.metric.summarization.loss import PGCriterion_REINFORCE
from src.eval import *
from torch.optim.lr_scheduler import LambdaLR
from tabulate import tabulate

def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    args = get_args()

    # for debug
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    # args = Argues('ruby.yml', 'summarization', 'unilang', 'deepcom', 'train_sl', 'source', True)  # train_sl
    # args = Argues('ruby.yml', 'summarization', 'unilang', 'deepcom', 'test', 'source', True)  # test
    LOGGER.info(args)

    config, dataset, = load_config_dataset(args, XlangDataloader, sBaseDataset, sbase_collate_fn, )

    # unilang-language
    src_lng = config['dataset'][args.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    model = build_model(config, DeepCom(config))

    # nohup python -u ./dataset/parse_key/main.py > key_100.log 2>&1 &
    if args.train_mode == 'train_sl':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim']) \
            (model.parameters(), config['sl']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(config['dataset'][args.dataset_type]['dataset_lng'])+
                                "_p{}".format(config['dataset']['portion']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.debug('save_dir: {}'.format(save_dir))
        sl_trainer = SLTrainer(config)

        lambda1 = lambda epoch: config['sl']['decay_coeff'] ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
        best_model = sl_trainer.train(model, unilang_dataset, lm_criterion, optimizer, SAVE_DIR=save_dir,scheduler=scheduler )

        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        for metric in ['bleu', 'cider']:
            model.load_state_dict(torch.load(best_model[metric], map_location=lambda storage, loc: storage))


            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                             metrics=['bleu', 'meteor', 'rouge', 'cider'])
            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
            tablefmt=config['common']['result_table_format'])))


            LOGGER.info("test_with_best_model_metric: {} ".format(best_model[metric]))
            print(headers)
            print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]])
            print("\n")

    elif args.train_mode == 'test':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider=\
            Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion)
        headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
        result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                              tablefmt=config['common']['result_table_format'])))

        print("\n")
        print(headers)
        print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]])


if __name__ == '__main__':
    main()
