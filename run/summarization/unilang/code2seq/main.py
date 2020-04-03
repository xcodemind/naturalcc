# -*- coding: utf-8 -*-
from run.util import *
from ncc.metric.base import *
from ncc.model.summarization.unilang import *
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
    # args_ = Argues('python.yml', 'summarization', 'unilang', 'code2seq', 'train_sl', 'source', True)
    # args_ = Argues('python.yml', 'summarization', 'unilang', 'code2seq', 'test', 'source', True)
    LOGGER.info(args_)

    args, dataset, = load_args_dataset(args_, XlangDataloader, sBaseDataset, sbase_collate_fn, )

    # unilang-language
    src_lng = args['dataset'][args_.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args_.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    LOGGER.info("args: {}".format(args))
    model = build_model(args, Code2Seq(args))

    # nohup python -u ./dataset/parse_key/main.py > key_100.log 2>&1 &
    if args_.train_mode == 'train_sl':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sl']['optim']) \
            (model.parameters(), args['sl']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(args['dataset'][args_.dataset_type]['dataset_lng'])+
                                "_k{}_p{}".format(args['dataset']['leaf_path_k'], args['dataset']['portion']),
                                args_.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.debug('save_dir: {}'.format(save_dir))
        sl_trainer = SLTrainer(args)
        best_model = sl_trainer.train(model, unilang_dataset, lm_criterion, optimizer, SAVE_DIR=save_dir, )

##
        for metric in ['bleu', 'cider']:
            model.load_state_dict(torch.load(best_model[metric], map_location=lambda storage, loc: storage))

            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                         metrics=['bleu', 'meteor', 'rouge', 'cider'])

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
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                     metrics=['bleu', 'meteor', 'rouge', 'cider'])

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
        if args_.dataset_type in ['all', 'target']:
            data_source = 'target'
        elif args_.dataset_type == 'source':
            data_source = 'source'
        trg_lng = args['dataset'][data_source]['dataset_lng'][0]
        unilang_dataset = dataset[data_source][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        model_filename = args['common']['init_weights']
        # Evaluator.case_study_eval(model, unilang_dataset['test'], dataset.token_dicts, model_filename=model_filename)
        Evaluator.case_study_eval_code2seq(model, unilang_dataset['test'], dataset.token_dicts, model_filename=model_filename)

if __name__ == '__main__':
    main()
