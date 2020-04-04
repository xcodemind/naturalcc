# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from ncc.metric.base import *
from ncc.models.summarization.unilang import *
from ncc.trainer.summarization.unilang import *
from ncc.trainer.summarization.xlang import *
from ncc.metric.summarization.loss import PGCriterion_REINFORCE
from ncc.eval import *
from tabulate import tabulate
from ncc.eval.evaluator import Evaluator


def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    args_ = get_args()

    # for debug
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    # pretrain
    # args_ = Argues('./python8ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'finetune', 'train_prt', 'source', True)
    # train_sl
    # args_ = Argues('./python8ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'finetune', 'train_ft', 'all', True)
    # args_ = Argues('ruby.yml', 'summarization', 'xlang', 'finetune', 'test', 'target', True)  # test
    LOGGER.info(args_)

    args, dataset, = load_args_dataset(args_, XlangDataloader, sBaseDataset, sbase_collate_fn, )
    model = build_model(args, MM2Seq(args))
    src_lng = args['dataset']['source']['dataset_lng'][0]
    src_dataset = dataset['source'][src_lng]
    trg_lng = args['dataset']['target']['dataset_lng'][0]
    trg_dataset = dataset['target'][trg_lng]

    if args_.train_mode == 'None' or args_.train_mode  is None :
        # step 1) pre-train
        args_.train_mode = 'train_prt'
        LOGGER.info('pretrain_on {}'.format(src_dataset))
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim'])(model.parameters(), args['sl']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'FT_prt_{}_ft_{}_p{}_pe{}'.format(args['dataset']['source']['dataset_lng'][0],
                                                             args['dataset']['target']['dataset_lng'][0],
                                                             args['dataset']['portion'],
                                                            args['finetune']['pretrain_epoch']), args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        ft_trainer = FTTrainer(args)
        ft_trainer.pretrain(model, src_dataset, lm_criterion, optimizer, SAVE_DIR=save_dir, )

        # step 2) train_ft
        args_.train_mode = 'train_ft'
        LOGGER.info(dataset)
        LOGGER.info(trg_lng)
        LOGGER.info('finetune_on {}'.format(trg_dataset, ))
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim'])(model.parameters(), args['sl']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'FT_prt_{}_ft_{}_p{}_pe{}'.format(args['dataset']['source']['dataset_lng'][0],
                                                             args['dataset']['target']['dataset_lng'][0],
                                                             args['dataset']['portion'],
                                                        args['finetune']['pretrain_epoch']), args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        ft_trainer = FTTrainer(args)
        best_model = ft_trainer.finetune(model, trg_dataset, lm_criterion, optimizer, SAVE_DIR=save_dir, )

        # step 3) testn need to specify model weights
        args_.train_mode = 'test'
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        for metric in args['testing']['metrics']:
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
                                                                      tablefmt=model.args['common'][
                                                                          'result_table_format'])))
                LOGGER.info("test_with_best_model_metric: {} ".format(best_model[metric]))
                print(headers)
                print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                             rouge1, rouge2, rouge3, rouge4, rougel, cider]])
                print("\n")

    elif args_.train_mode == 'test':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider =\
            Evaluator.summarization_eval(model, trg_dataset['test'], dataset.token_dicts, lm_criterion,
                                         metrics=['bleu', 'meteor', 'rouge', 'cider'],)

        headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
        result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                              tablefmt=model.args['common'][
                                                                  'result_table_format'])))
        print("\n")
        print(headers)
        print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]])

    elif args_.train_mode == 'case_study':
        if args_.dataset_type in ['all', 'target'] :
            data_source = 'target'
        elif args_.dataset_type == 'source':
            data_source = 'source'
        trg_lng = args['dataset'][data_source]['dataset_lng'][0]
        unilang_dataset = dataset[data_source][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        model_filename = args['common']['init_weights']
        Evaluator.case_study_eval(model, unilang_dataset['test'], dataset.token_dicts, model_filename=model_filename)

if __name__ == '__main__':
    main()
