# -*- coding: utf-8 -*-
from run.util import *
from ncc.metric.base import *
from ncc.model.summarization.unilang import *
from ncc.trainer.summarization.unilang import *
from ncc.utils.util_file  import load_args8yml
from tabulate import tabulate
from ncc.eval.evaluator import Evaluator
from ncc.dataset.base.ast_attendgru_dataset import AstAttendGruDataset, ast_attendgru_collate_fn


def main():
    args_ = get_args()
    LOGGER.info(args_)

    args  = load_args8yml(args_)
    args = run_init(args_.yaml,args=args)
    args, dataset, = load_args_dataset_ast_attendgru(
        args_, XlangDataloader, AstAttendGruDataset, ast_attendgru_collate_fn, args=args)

    # unilang-language
    src_lng = args['dataset']['source']['dataset_lng'][0]
    unilang_dataset = dataset['source'][src_lng]
    LOGGER.info(unilang_dataset)

    model = build_model(args,   AstAttendGru(args,unilang_dataset.token_dicts['comment']))
    if args_.train_mode == 'train_sl':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(args['dataset'][args_.dataset_type]['dataset_lng']), args_.train_mode)
        LOGGER.debug('save_dir: {}'.format(save_dir))

        sl_trainer = AstAttendGruSLTrainer(args, model, unilang_dataset )
        sl_trainer.train( lm_criterion )
    elif args_.train_mode == 'test':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider =\
        Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                     model_filename=args['common']['init_weights'])

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
