# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from ncc.metric.base import *
from ncc.model.summarization.unilang import *
from ncc.trainer.summarization.unilang import *
from ncc.trainer.summarization.xlang import *
from ncc.metric.summarization.loss import PGCriterion_REINFORCE
from ncc.eval import *
from ncc.utils.constants import METRICS
from tabulate import tabulate

'''
DTRL, 可能需要调整的参数
# device: GPU id
# init_weights: 用于sc/test
# dataset_dir: 数据集目录
# save_dir: 保存模型目录
# portion: 旨在ruby的train数据集有效
# source/target: 根据任务调整
# code_modalities: 根据任务调整
# batch_size: 根据显存调整
# metrics: 测试务必全选
# dtrl/kd/maml: 根据自己的要求调整
'''


def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    args = get_args()

    # for debug
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')

    # train_dtrl_sl
    # nohup python -u ./run/summarization/xlang/dtrl/main.py > ./run/summarization/xlang/dtrl/train_dtrl_sl.log 2>&1 &
    # args = Argues('./python8ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'dtrl', 'train_dtrl_sl', 'all', True)

    # train_dtrl_sc
    # args = Argues('./javascript8ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'dtrl', 'train_dtrl_sc', 'target', True)

    # test
    # python -u ./run/summarization/xlang/dtrl/main.py
    # args = Argues('./javascript8ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'dtrl', 'test', 'all', True)
    LOGGER.info(args)


    config, dataset, = load_config_dataset(args, XlangDataloader, sBaseDataset, lambda tensor: tensor)
    model = build_model(config, MM2Seq(config))

    trg_lng = config['dataset']['target']['dataset_lng'][0]
    trg_dataset = dataset['target'][trg_lng]

    if args.train_mode == 'train_dtrl_sl' or args.train_mode == 'None' or args.train_mode is None:
        # args.train_mode = 'train_dtrl_sl'
        LOGGER.info('DTRL train_on {}'.format(dataset))
        dtrl_criterion = TRLLoss(device=config['common']['device'] is not None,
                                 eta=config['dtrl']['eta'], zeta=config['dtrl']['zeta'])

        optimizer = getattr(torch.optim, config['dtrl']['optim'])(model.parameters(), config['dtrl']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'DTRL_{}_p{}'.format('-'.join(sorted(config['dataset']['source']['dataset_lng'])),
                                                     config['dataset']['portion']),
                                args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.debug('save_dir: {}'.format(save_dir))
        dtrl_trainer = DTRLTrainer(config)
        best_model = dtrl_trainer.train(model, dataset, dtrl_criterion, optimizer, SAVE_DIR=save_dir, )



    if args.train_mode == 'None' or args.train_mode is None:
        # elif args.train_mode == 'test':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        for metric in ['bleu', 'cider']:
            model.load_state_dict(torch.load(best_model[metric], map_location=lambda storage, loc: storage))
            LOGGER.info('test on {}, model weights of best {} from {}'.format(trg_dataset, metric, best_model[metric]))

            # Validation on each epoch
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, trg_dataset['test'], dataset.token_dicts, lm_criterion,
                                             collate_func=sbase_collate_fn, metrics=METRICS)



            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                  tablefmt=model.config['common']['result_table_format'])))

            LOGGER.info("test_with_best_model_metric: {} ".format(best_model[metric]))
            print(headers)
            print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                         rouge1, rouge2, rouge3, rouge4, rougel, cider]])
            print("\n")

    if  args.train_mode ==  'test':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        # Validation on each epoch
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(model, trg_dataset['test'], dataset.token_dicts, lm_criterion,
                                         collate_func=sbase_collate_fn, metrics=METRICS)
        headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
        result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                              tablefmt=model.config['common'][
                                                                  'result_table_format'])))

        print("\n")
        print(headers)
        print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                     rouge1, rouge2, rouge3, rouge4, rougel, cider]])

# elif args.train_mode == 'train_dtrl_sc':
#     trg_lng = config['dataset']['target']['dataset_lng'][0]
#     unilang_dataset = dataset[args.dataset_type][trg_lng]
#     LOGGER.info(unilang_dataset)
#
#     pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
#     lm_criterion = LMLoss(device=config['common']['device'] is not None, )
#     optimizer = getattr(torch.optim, config['sc']['optim'])(model.parameters(), config['sc']['lr'])
#     save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
#                             'DTRL_{}_p{}'.format('-'.join(sorted(config['dataset']['source']['dataset_lng'])),
#                                                  config['dataset']['portion']),
#                             args.train_mode)
#     os.makedirs(save_dir, exist_ok=True)
#     LOGGER.info('save_dir: {}'.format(save_dir))
#     sc_trainer = SCTrainer(config)
#     sc_trainer.train(model, unilang_dataset, lm_criterion, pg_criterion, optimizer,
#                      config['sc']['reward_func'], SAVE_DIR=save_dir, )


if __name__ == '__main__':
    main()
