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
from tabulate import tabulate

def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    args = get_args()

    # for debug
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    # pretrain
    # args = Argues('./python8ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'finetune', 'train_prt', 'source', True)
    # train_sl
    # args = Argues('./python8ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'finetune', 'train_ft', 'all', True)
    # args = Argues('ruby.yml', 'summarization', 'xlang', 'finetune', 'test', 'target', True)  # test
    LOGGER.info(args)

    config, dataset, = load_config_dataset(args, XlangDataloader, sBaseDataset, sbase_collate_fn, )
    model = build_model(config, MM2Seq(config))
    src_lng = config['dataset']['source']['dataset_lng'][0]
    src_dataset = dataset['source'][src_lng]
    trg_lng = config['dataset']['target']['dataset_lng'][0]
    trg_dataset = dataset['target'][trg_lng]

    if args.train_mode == 'None' or args.train_mode  is None :
        # step 1) pre-train
        args.train_mode = 'train_prt'
        LOGGER.info('pretrain_on {}'.format(src_dataset))
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim'])(model.parameters(), config['sl']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'FT_prt_{}_ft_{}_p{}_pe{}'.format(config['dataset']['source']['dataset_lng'][0],
                                                             config['dataset']['target']['dataset_lng'][0],
                                                             config['dataset']['portion'],
                                                            config['finetune']['pretrain_epoch']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        ft_trainer = FTTrainer(config)
        ft_trainer.pretrain(model, src_dataset, lm_criterion, optimizer, SAVE_DIR=save_dir, )

        # step 2) train_ft
        args.train_mode = 'train_ft'
        LOGGER.info(dataset)
        LOGGER.info(trg_lng)
        LOGGER.info('finetune_on {}'.format(trg_dataset, ))
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim'])(model.parameters(), config['sl']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'FT_prt_{}_ft_{}_p{}_pe{}'.format(config['dataset']['source']['dataset_lng'][0],
                                                             config['dataset']['target']['dataset_lng'][0],
                                                             config['dataset']['portion'],
                                                        config['finetune']['pretrain_epoch']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        ft_trainer = FTTrainer(config)
        best_model = ft_trainer.finetune(model, trg_dataset, lm_criterion, optimizer, SAVE_DIR=save_dir, )

        # step 3) testn need to specify model weights
        args.train_mode = 'test'
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        for metric in config['testing']['metrics']:
            if best_model.__contains__(metric):
                model.load_state_dict(torch.load(best_model[metric], map_location=lambda storage, loc: storage))
                LOGGER.info('test on {}, model_weights_of_best {} from {}'.format(trg_dataset, metric, best_model[metric]))
                bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider=\
                    Evaluator.summarization_eval(model, trg_dataset['test'], dataset.token_dicts, lm_criterion,
                                                 metrics=['bleu', 'meteor', 'rouge', 'cider'],)



                headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
                result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                       rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
                LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                      tablefmt=model.config['common'][
                                                                          'result_table_format'])))
                LOGGER.info("test_with_best_model_metric: {} ".format(best_model[metric]))
                print(headers)
                print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                             rouge1, rouge2, rouge3, rouge4, rougel, cider]])
                print("\n")

    elif args.train_mode == 'test':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        # for metric in config['testing']['metrics']:
            # model.load_state_dict(torch.load(best_model[metric], map_location=lambda storage, loc: storage))
            # LOGGER.info('test on {}, model weights of best {} from {}'.format(trg_dataset, metric, best_model[metric]))
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider =\
            Evaluator.summarization_eval(model, trg_dataset['test'], dataset.token_dicts, lm_criterion,
                                         metrics=['bleu', 'meteor', 'rouge', 'cider'],)

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

    elif args.train_mode == 'case_study':
        if args.dataset_type in ['all', 'target'] :
            data_source = 'target'
        elif args.dataset_type == 'source':
            data_source = 'source'
        trg_lng = config['dataset'][data_source]['dataset_lng'][0]
        unilang_dataset = dataset[data_source][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        model_filename = config['common']['init_weights']
        Evaluator.case_study_eval(model, unilang_dataset['test'], dataset.token_dicts, model_filename=model_filename)

if __name__ == '__main__':
    main()
