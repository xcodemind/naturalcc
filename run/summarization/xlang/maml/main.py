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
from ncc.eval.evaluator import Evaluator


def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    # args_ = get_args()

    # for debug
    Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_ppjj_r.log 2>&1 &
    # args_ = Argues('config.yml', 'summarization', 'xlang', 'maml', 'train_maml', 'all', True)  # train_sl

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_ppjj_ft_p0.log 2>&1 &
    # args_ = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'train_maml_ft', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/zero_shot_test.log 2>&1 &
    # args_ = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'train_select', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/case_study.log 2>&1 &
    # args_ = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'case_study', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_ppjj_sc_p0.log 2>&1 &
    # args_ = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'train_maml_sc', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_test.log 2>&1 &
    args_ = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'test', 'target',
                  True)  # test
    LOGGER.info(args_)

    args, dataset, = load_args_dataset(args_, XlangDataloader, sBaseDataset, sbase_collate_fn, )
    model = build_model(args, MM2Seq(args))
    LOGGER.info(model)

    trg_lng = args['dataset']['target']['dataset_lng'][0]  # ruby
    trg_dataset = dataset['target'][trg_lng]

    if args_.train_mode == 'train_maml':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim'])(model.parameters(), args['sl']['lr'])
        meta_optimizer = getattr(torch.optim, args['maml']['meta_optim']) \
            (model.parameters(), args['maml']['meta_lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'maml_{}_{}_{}({})-{}({})_{}_{}'.format(
                                    '-'.join(sorted(args['dataset']['source']['dataset_lng'])),
                                    '-'.join(sorted(args['dataset']['target']['dataset_lng'])),
                                    args['sl']['optim'], args['sl']['lr'],
                                    args['maml']['meta_optim'], args['sl']['meta_lr'],
                                    args['maml']['meta_train_size'],
                                    args['maml']['meta_val_size'],
                                ),
                                args_.train_mode)

        # save_dir = None
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.debug('save_dir: {}'.format(save_dir))
        maml_trainer = MAMLTrainer(args)
        maml_trainer.train(model, dataset, lm_criterion, optimizer, meta_optimizer, SAVE_DIR=save_dir, )

    elif args_.train_mode == 'train_maml_ft':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim'])(model.parameters(), args['sl']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'],
                                'maml_ft_{}_{}({})'.format(args['dataset']['portion'], args['sl']['optim'],
                                                           args['sl']['lr']))
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.debug('save_dir: {}'.format(save_dir))

        ft_trainer = FTTrainer(args)
        best_model = ft_trainer.finetune(model, dataset['target'][trg_lng], lm_criterion, optimizer,
                                         SAVE_DIR=save_dir)
        LOGGER.info(best_model)

        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        for metric in ['bleu', 'cider', ]:
            if metric in best_model:
                checkpoint = torch.load(best_model[metric], map_location=lambda storage, loc: storage)
                model.load_state_dict(checkpoint)
                LOGGER.info(
                    'test on {}, model weights of best {} from {}'.format(trg_dataset, metric, best_model[metric]))
                bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                    Evaluator.summarization_eval(model, trg_dataset['test'], dataset.token_dicts, lm_criterion,
                                                 metrics=METRICS, )
                headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
                result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                       rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
                LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                      tablefmt=args['common'][
                                                                          'result_table_format'])))
        LOGGER.info('=' * 20)

    elif args_.train_mode == 'train_select':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        maml_trainer = MAMLTrainer(args)
        for epoch in range(10, 4168, 10):
            pt_filename = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/maml_java-javascript-php-python_ruby_Adam(0.0004)_SGD(0.001)-10-1.new/train_maml/tok8path-bs128-Adam(0.0004)-mSGD(0.001)-EPOCH10-1-{}.pt'.format(
                epoch)
            LOGGER.info('load from {}'.format(pt_filename))
            checkpoint = torch.load(pt_filename, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)
            optimizer = getattr(torch.optim, args['sl']['optim'])(model.parameters(), args['sl']['lr'])
            maml_trainer.finetune(model, dataset['target'][trg_lng], lm_criterion, optimizer)


    elif args_.train_mode == 'test':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        trg_lng = args['dataset']['target']['dataset_lng'][0]
        unilang_dataset = dataset['target'][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                         metrics=METRICS)
        headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
        result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                              tablefmt=args['common']['result_table_format'])))

    elif args_.train_mode == 'train_maml_sc':
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sc']['optim']) \
            (model.parameters(), args['sc']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(args['dataset']['source']['dataset_lng']) +
                                "_p{}_bi{}".format(args['dataset']['portion'],
                                                   args['training']['rnn_bidirectional']), args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        trg_lng = args['dataset']['target']['dataset_lng'][0]
        unilang_dataset = dataset['target'][trg_lng]

        sc_trainer = SCTrainer(args)
        sc_trainer.train(model, unilang_dataset, lm_criterion, pg_criterion, optimizer,
                         args['sc']['reward_func'], SAVE_DIR=save_dir, )


    elif args_.train_mode == 'case_study':
        trg_lng = args['dataset']['target']['dataset_lng'][0]
        unilang_dataset = dataset['target'][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        model_filename = args['common']['init_weights']
        Evaluator.case_study_eval(model, unilang_dataset['test'], dataset.token_dicts, model_filename=model_filename)


    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
