# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from ncc.metric.base import *
from ncc.model.summarization.unilang import *
from ncc.trainer.summarization.xlang import *
from ncc.trainer.summarization.unilang import *
from ncc.metric.summarization.loss import PGCriterion_REINFORCE
from ncc.eval import *
from ncc.utils.util_file  import load_args8yml
from tabulate import tabulate
from ncc.eval.evaluator import Evaluator


def main():
    args_ = get_args()
    LOGGER.info(args_)

    args = load_args8yml(args_)
    args = run_init_kd(args_.yaml, args=args)

    args, dataset = load_args_dataset_kd(args_,  base_dataset=sBaseDataset, args=args)

    args = get_model_path(args_, args)
    model = build_model_kd(args, args_.dataset_type, MM2Seq(args))


    assert args_.dataset_type == 'source'

    if args_.train_mode in ['train_sl_ft','train_sc_ft','test']:
        # unilang-language
        src_lng = args['dataset'][args_.dataset_type]['dataset_lng'][0]
        unilang_dataset = dataset[args_.dataset_type][src_lng]

    if args_.train_mode in ['train_sl']:
        if args['kd']['distill']:
            lm_criterion = LMCriterionLabelSmoothKD(device=args['common']['device'] is not None,
                            label_smooth_rate=args['kd']['label_smooth_rate'], distill_temp=args['kd']['distill_temp'])
        else:
            lm_criterion = LMCriterionLabelSmooth(device=args['common']['device'] is not None,
                                                  label_smooth_rate=args['kd']['label_smooth_rate'])
         # optim = torch.optim.Adam(model.parameters(), args[args_.dataset_type]['lr'])
        optimizer = getattr(torch.optim, args['sl']['optim']) \
            (model.parameters(), args['sl']['lr'])
        print('created model sucessfully....................')
        sl_trainer = KDSLTrainer(args_, args, model, dataset )
        sl_trainer.train(lm_criterion, optimizer)

    elif args_.train_mode == 'train_sl_ft':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim']) \
            (model.parameters(), args['sl']['lr'])
        # save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
        #                         '-'.join(args['dataset'][args_.dataset_type]['dataset_lng']), args_.train_mode)
        #
        # if args['training']['tuning']:
        #     save_dir = os.path.join(save_dir, args['training']['tuning'])
        # os.makedirs(save_dir, exist_ok=True)
        # LOGGER.info('save_dir: {}'.format(save_dir))
        sl_trainer = SLTrainer(args)
        model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-orin-{}-bi{}'.format(
            '8'.join(args['training']['code_modalities']),
            args['training']['batch_size'],
            args['sl']['lr'],
            args['training']['attn_type'],
            args['training']['pointer'], args['sl']['oriname2finetune'], args['training']['rnn_bidirectional'])
        sl_trainer.train(model, unilang_dataset, lm_criterion,
                         optimizer, SAVE_DIR=args['dataset']['save_dir'],
                         model_name_prefix= model_name_prefix )

    elif args_.train_mode == 'train_sc_ft':
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sc']['optim']) \
            (model.parameters(), args['sc']['lr'])
        # save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
        #                         '-'.join(args['dataset']['source']['dataset_lng']), args_.train_mode)
        # os.makedirs(save_dir, exist_ok=True)
        # LOGGER.info('save_dir: {}'.format(save_dir))
        sc_trainer = SCTrainer(args)
        model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-orin-{}-bi{}'.format(
            '8'.join(args['training']['code_modalities']),
            args['training']['batch_size'],
            args['sl']['lr'],
            args['training']['attn_type'],
            args['training']['pointer'], args['sc']['oriname2finetune'], args['training']['rnn_bidirectional'])
        sc_trainer.train(model, unilang_dataset, lm_criterion, pg_criterion, optimizer,
                         args['sc']['reward_func'], SAVE_DIR=args['dataset']['save_dir'],
                         model_name_prefix=model_name_prefix )

    elif args_.train_mode == 'test':
        criterion = LMLoss(device=args['common']['device'] is not None, )
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts,
                                     criterion =criterion, model_filename=args['model_path'])
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

    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
