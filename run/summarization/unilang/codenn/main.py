# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from run.util import *
from ncc.metric.base import *
from ncc.models.summarization.unilang import *
from ncc.trainer.summarization.unilang import *
# from ncc.trainer.summarization.xlang import *

from ncc.eval import *
from ncc.utils.util_file  import load_args8yml
from tabulate import tabulate
from ncc.eval.evaluator import Evaluator


def main():
 #python run/main.py --yaml ruby.yml --task summarization --lang_mode unilang --method_name codenn --train_mode test --dataset_type source --multi_processing 0 --debug 0
 #python run/main.py --yaml ruby.yml --task summarization --lang_mode unilang --method_name codenn --train_mode train_sl --dataset_type source --multi_processing 0 --debug 0

    args_ = get_args()
    LOGGER.info(args_)

    # args  = load_args8yml(args)
    # args = run_init(args.yaml,args=args)

    args, dataset, = load_args_dataset(args_, XlangDataloader, sBaseDataset, sbase_collate_fn)

    # unilang-language
    src_lng = args['dataset']['source']['dataset_lng'][0]
    unilang_dataset = dataset['source'][src_lng]
    LOGGER.info(unilang_dataset)


    model = build_model(args, CodeNN(args, unilang_dataset.token_dicts['comment']))


    if args_.train_mode == 'train_sl':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(args['dataset'][args_.dataset_type]['dataset_lng'])+
                                "_p{}".format(args['dataset']['portion'] ), args_.train_mode)
        LOGGER.debug('save_dir: {}'.format(save_dir))
        os.makedirs(save_dir, exist_ok=True)
        sl_trainer = CodeNNSLTrainer(args, model, unilang_dataset )
        sl_trainer.train(lm_criterion, SAVE_DIR=save_dir)


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
# model: IModel, data_loader: DataLoader, token_dicts: TokenDicts, criterion: BaseLoss,
    #                            model_filename
    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
