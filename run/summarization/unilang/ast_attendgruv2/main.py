# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from ncc.metric.base import *
from ncc.model.summarization.unilang import *
from ncc.trainer.summarization.unilang import *


from ncc.eval import *
from ncc.utils.util_file  import load_args8yml
from tabulate import tabulate

def main():

    args = get_args()
    LOGGER.info(args)

    config  = load_args8yml(args)
    config = run_init(args.yaml,config=config)


    config, dataset, = load_config_dataset_ast_attendgru(
        args, XlangDataloader, AstAttendGruDataset, ast_attendgruv2_collate_fn, config=config)

    # unilang-language
    src_lng = config['dataset']['source']['dataset_lng'][0]
    unilang_dataset = dataset['source'][src_lng]
    LOGGER.info(unilang_dataset)


    model = build_model(config,   AstAttendGruV2(config,unilang_dataset.token_dicts['comment']))


    if args.train_mode == 'train_sl':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(config['dataset'][args.dataset_type]['dataset_lng']), args.train_mode)
        LOGGER.debug('save_dir: {}'.format(save_dir))

        sl_trainer = AstAttendGruSLTrainer(config, model, unilang_dataset )
        sl_trainer.train( lm_criterion )


    elif args.train_mode == 'test':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider =\
        Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                     model_filename=config['common']['init_weights'])

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
# model: IModel, data_loader: DataLoader, token_dicts: TokenDicts, criterion: BaseLoss,
    #                            model_filename
    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
