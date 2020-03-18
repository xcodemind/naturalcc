# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from src.metric.base import *
from src.model.summarization.unilang import *
from src.trainer.summarization.xlang import *
from src.trainer.summarization.unilang import *
from src.metric.summarization.loss import PGCriterion_REINFORCE
from src.eval import *
from src.utils.util_file  import load_args8yml
from tabulate import tabulate

def main():
    args = get_args()
    LOGGER.info(args)

    config  = load_args8yml(args)
    config = run_init_kd(args.yaml,config=config)

    config,  dataset    = load_config_dataset_kd(args,  base_dataset=sBaseDataset,config=config)

    config = get_model_path(args, config)
    model = build_model_kd(config, args.dataset_type, MM2Seq(config ))


    assert args.dataset_type == 'source'

    if args.train_mode in [ 'train_sl_ft','train_sc_ft','test']:
        # unilang-language
        src_lng = config['dataset'][args.dataset_type]['dataset_lng'][0]
        unilang_dataset = dataset[args.dataset_type][src_lng]

    if args.train_mode in [ 'train_sl']:
        if config['kd']['distill']:
            lm_criterion = LMCriterionLabelSmoothKD(device=config['common']['device'] is not None,
                            label_smooth_rate=config['kd']['label_smooth_rate'],distill_temp=config['kd']['distill_temp'])
        else:
            lm_criterion = LMCriterionLabelSmooth(device=config['common']['device'] is not None,
                                                  label_smooth_rate=config['kd']['label_smooth_rate'])
         # optim = torch.optim.Adam(model.parameters(), config[args.dataset_type]['lr'])
        optimizer = getattr(torch.optim, config['sl']['optim']) \
            (model.parameters(), config['sl']['lr'])
        print('created model sucessfully....................')
        sl_trainer = KDSLTrainer(args,config,model,dataset )
        sl_trainer.train(lm_criterion, optimizer)

    elif args.train_mode == 'train_sl_ft':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim']) \
            (model.parameters(), config['sl']['lr'])
        # save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
        #                         '-'.join(config['dataset'][args.dataset_type]['dataset_lng']), args.train_mode)
        #
        # if config['training']['tuning']:
        #     save_dir = os.path.join(save_dir, config['training']['tuning'])
        # os.makedirs(save_dir, exist_ok=True)
        # LOGGER.info('save_dir: {}'.format(save_dir))
        sl_trainer = SLTrainer(config)
        model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-orin-{}-bi{}'.format(
            '8'.join(config['training']['code_modalities']),
            config['training']['batch_size'],
            config['sl']['lr'],
            config['training']['attn_type'],
            config['training']['pointer'],config['sl']['oriname2finetune'],config['training']['rnn_bidirectional'])
        sl_trainer.train(model, unilang_dataset, lm_criterion,
                         optimizer, SAVE_DIR=config['dataset']['save_dir'],
                         model_name_prefix= model_name_prefix )

    elif args.train_mode == 'train_sc_ft' :
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sc']['optim']) \
            (model.parameters(), config['sc']['lr'])
        # save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
        #                         '-'.join(config['dataset']['source']['dataset_lng']), args.train_mode)
        # os.makedirs(save_dir, exist_ok=True)
        # LOGGER.info('save_dir: {}'.format(save_dir))
        sc_trainer = SCTrainer(config)
        model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-orin-{}-bi{}'.format(
            '8'.join(config['training']['code_modalities']),
            config['training']['batch_size'],
            config['sl']['lr'],
            config['training']['attn_type'],
            config['training']['pointer'],config['sc']['oriname2finetune'],config['training']['rnn_bidirectional'])
        sc_trainer.train(model, unilang_dataset, lm_criterion, pg_criterion, optimizer,
                         config['sc']['reward_func'], SAVE_DIR=config['dataset']['save_dir'],
                         model_name_prefix=model_name_prefix )



    elif args.train_mode == 'test':

        criterion = LMLoss(device=config['common']['device'] is not None, )
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(model,  unilang_dataset['test'], dataset.token_dicts,
                                     criterion =criterion ,model_filename=config['model_path'] )
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

    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
