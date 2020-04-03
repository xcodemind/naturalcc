# -*- coding: utf-8 -*-
import sys
import os
# sys.path.append('.')
import torch
import torch.nn as nn
from run.util import *
from ncc.metric.base import *
from ncc.model.summarization.unilang import *
from ncc.trainer.summarization.unilang import *
from ncc.trainer.summarization.xlang import *
from ncc.metric.summarization.loss import PGCriterion_REINFORCE
from ncc.eval import *
from tabulate import tabulate

def main():
    # python -u ./run/summarization/unilang/mm2seq/mm2seq.py --yaml ./finetune/ruby-python.yml --task summarization
    # --lang_mode unilang --method_name mm2seq --args.train_mode train_sl --load_src True --load_trg False
    # args = get_args()

    # for debug
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    # args = Argues('ruby.yml', 'summarization', 'unilang', 'mm2seq', 'train_sc', 'source', True)  # train_sl
    # args = Argues('ruby.yml', 'summarization', 'unilang', 'mm2seq', 'test', 'source', True)  # test
    # args = Argues('ruby.yml', 'summarization', 'unilang', 'mm2seq', 'train_maml', 'source', True)  # maml
    # args = Argues('python.yml', 'summarization', 'unilang', 'mm2seq', 'train_sl', 'source', True)  # train_sl
    args_ = get_args()

    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False

    args, dataset, = load_args_dataset(args_, XlangDataloader, sBaseDataset, sbase_collate_fn, )

    # SAVE_DIR = get_save_dir(args, args_)
    # LOGGER.debug(SAVE_DIR)

    # unilang-language
    src_lng = args['dataset'][args.dataset_type]['dataset_lng'][0]
    unilang_dataset = dataset[args.dataset_type][src_lng]
    LOGGER.info(unilang_dataset)

    # # xlang-language
    # xlang_dataset = dataset[args.dataset_type]
    # LOGGER.info(xlang_dataset)

    LOGGER.info("args: {}".format(args))
    model = build_model(args, MM2Seq(args))
    # assert model.encoder.tok_encoder.wemb.embedding.weight.size(0) == args['code_token_num']
    # assert model.decoder.wemb.embedding.weight.size(0) == args['comment_token_num']
    # assert model.decoder.linear.weight.size(0) == args['comment_token_num']

    if args.train_mode == 'train_sl':
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )

        optimizer = getattr(torch.optim, args['sl']['optim']) \
            (model.parameters(), args['sl']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
                '-'.join(args['dataset'][args.dataset_type]['dataset_lng'])+
                                "_p{}_bi{}".format(args['dataset']['portion'],args['training']['rnn_bidirectional']), args.train_mode)

        if args['training']['tuning']:
            save_dir = os.path.join(save_dir, args['training']['tuning'])
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        sl_trainer = SLTrainer(args)
        best_model = sl_trainer.train(model, unilang_dataset, lm_criterion, optimizer, SAVE_DIR=save_dir, )

    elif args.train_mode == 'train_pg':
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['pg']['optim']) \
            (model.parameters(), args['pg']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
            '-'.join(args['dataset']['source']['dataset_lng'])+
                                "_p{}_bi{}".format(args['dataset']['portion'],args['training']['rnn_bidirectional']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        pg_trainer = PGTrainer(args)
        pg_trainer.train(model, unilang_dataset, lm_criterion, pg_criterion, optimizer,
                         args['pg']['reward_func'], SAVE_DIR=save_dir, )

    elif args.train_mode == 'train_sc':
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        optimizer = getattr(torch.optim, args['sc']['optim']) \
            (model.parameters(), args['sc']['lr'])
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
            '-'.join(args['dataset']['source']['dataset_lng'])+
                                "_p{}_bi{}_rf{}".format(args['dataset']['portion'],
                                args['training']['rnn_bidirectional'],  args['sc']['reward_func']),
                                args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        sc_trainer = SCTrainer(args)
        best_model = sc_trainer.train(model, unilang_dataset, lm_criterion, pg_criterion, optimizer,
                         args['sc']['reward_func'], SAVE_DIR=save_dir, )

    elif args.train_mode == 'train_critic':
        critic = build_critic(args, MMCritic(args))
        critic_criterion = torch.nn.MSELoss(reduction='none').cuda()  # TODO: to optimized like LMLoss
        critic_optimizer = getattr(torch.optim, args['ac']['optim_critic']) \
            (critic.parameters(), args['ac']['lr_critic'])  # [args.dataset_type]
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
         '-'.join(args['dataset']['source']['dataset_lng'])+
                                "_p{}_bi{}".format(args['dataset']['portion'],args['training']['rnn_bidirectional']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        critic_trainer = CriticTrainer(args)
        critic_trainer.train(model, critic, unilang_dataset, critic_criterion, critic_optimizer, SAVE_DIR=save_dir, )

    elif args.train_mode == 'train_ac':
        optimizer = getattr(torch.optim, args['ac']['optim']) \
            (model.parameters(), args['ac']['lr'])
        critic = build_critic(args, MMCritic(args))
        critic_optimizer = getattr(torch.optim, args['ac']['optim_critic']) \
            (critic.parameters(), args['ac']['lr_critic'])
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        critic_criterion = torch.nn.MSELoss().cuda()  # TODO: to optimized like LMLoss
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
         '-'.join(args['dataset']['source']['dataset_lng'])+
                                "_p{}_bi{}".format(args['dataset']['portion'],args['training']['rnn_bidirectional']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        ac_trainer = ACTrainer(args)
        ac_trainer.train(model, critic, unilang_dataset, pg_criterion, lm_criterion, critic_criterion, optimizer,
                         critic_optimizer, SAVE_DIR=save_dir, )

    elif args.train_mode == 'train_disc':
        disc = build_disc(args, args.dataset_type, MMDiscriminator(args))
        disc_criterion = torch.nn.NLLLoss(reduction='sum')  # TODO: to optimized like LMLoss
        disc_optimizer = getattr(torch.optim, args['gan']['optim_disc']) \
            (disc.parameters(), args['gan']['lr_disc'])  # [args.dataset_type]
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
             '-'.join(args['dataset']['source']['dataset_lng'])+
                                "_p{}_bi{}".format(args['dataset']['portion'],args['training']['rnn_bidirectional']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        disc_trainer = DiscTrainer(args)
        disc_trainer.generate_training_pairs(model, unilang_dataset,
                                             filepath=os.path.join(args['gan']['comment_fake_path'],
                                                                   '{}_fake.txt'.format(
                                                                       args['dataset']['source']['dataset_lng'][0])))
        assert False
        disc_trainer.train(model, disc, unilang_dataset, disc_criterion, disc_optimizer, SAVE_DIR=save_dir, )

    elif args.train_mode == 'train_gan':
        optimizer = getattr(torch.optim, args['gan']['optim']) \
            (model.parameters(), args['gan']['lr'])
        disc = build_disc(args, args.dataset_type, MMDiscriminator(args))
        disc_optimizer = getattr(torch.optim, args['gan']['optim_disc']) \
            (disc.parameters(), args['gan']['lr_disc'])
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        disc_criterion = torch.nn.MSELoss().cuda()  # TODO: to optimized like LMLoss
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
            '-'.join(args['dataset']['source']['dataset_lng'])+
                                "_p{}_bi{}".format(args['dataset']['portion'],args['training']['rnn_bidirectional']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        gan_trainer = GANTrainer(args)
        gan_trainer.train(model, disc, unilang_dataset, pg_criterion, lm_criterion, disc_criterion, optimizer,
                          disc_optimizer, SAVE_DIR=save_dir, )

    elif args.train_mode == 'train_reward_model':
        pass

    elif args.train_mode == 'train_arel':
        optimizer = getattr(torch.optim, args['arel']['optim']) \
            (model.parameters(), args['arel']['lr'])
        reward_model = build_reward_model(args, args.dataset_type, MMRewardModel(args))
        reward_model_optimizer = getattr(torch.optim, args['arel']['optim_reward_model']) \
            (reward_model.parameters(), args['arel']['lr_reward_model'])
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        reward_model_criterion = torch.nn.MSELoss().cuda()  # TODO: to optimized like LMLoss
        save_dir = os.path.join(args['dataset']['save_dir'], model.__class__.__name__.lower(),
         '-'.join(args['dataset']['source']['dataset_lng'])+
                                "_p{}_bi{}".format(args['dataset']['portion'],args['training']['rnn_bidirectional']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))
        arel_trainer = ARELTrainer(args)
        arel_trainer.train(model, reward_model, unilang_dataset, pg_criterion, lm_criterion, reward_model_criterion,
                           optimizer,
                           reward_model_optimizer, SAVE_DIR=save_dir, )

    # elif args.train_mode == 'train_maml':
    #     lm_criterion = LMLoss(device=args['common']['device'] is not None, )
    #     optimizer = getattr(torch.optim, args[args.dataset_type]['optim']) \
    #         (model.parameters(), args[args.dataset_type]['lr'])
    #     meta_optimizer = getattr(torch.optim, args[args.dataset_type]['meta_optim']) \
    #         (model.parameters(), args[args.dataset_type]['meta_lr'])
    #     maml_trainer = MAMLTrainer()
    #     maml_trainer.train(model, xlang_dataset, lm_criterion, optimizer, meta_optimizer, SAVE_DIR=None, )

    # elif args.train_mode == 'train_ft':
    #     lm_criterion = LMLoss(device=args['common']['device'] is not None, )
    #     optimizer = getattr(torch.optim, args[args.dataset_type]['optim']) \
    #         (model.parameters(), args[args.dataset_type]['lr'])
    #     ft_trainer = FTTrainer()
    #     ft_trainer.train(model, unilang_dataset, lm_criterion, optimizer, SAVE_DIR=None, )


    #  test with best model
    if args.train_mode in [ 'train_sl' , 'train_sc' ]:
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        for metric in ['bleu', 'cider']:
            model.load_state_dict(torch.load(best_model[metric], map_location=lambda storage, loc: storage))


            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                             metrics=['bleu', 'meteor', 'rouge', 'cider'])
            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
            tablefmt=args['common']['result_table_format'])))


            LOGGER.info("test_with_best_model_metric: {} ".format(best_model[metric]))
            print(headers)
            print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]])
            print("\n")

    elif args.train_mode == 'test' :
        lm_criterion = LMLoss(device=args['common']['device'] is not None, )
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(model, unilang_dataset['valid'], dataset.token_dicts, lm_criterion,
                                         metrics=['bleu', 'meteor', 'rouge', 'cider'])
        headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
        result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
        tablefmt=args['common']['result_table_format'])))

        print("\n")
        print(headers)
        print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                               rouge1, rouge2, rouge3, rouge4, rougel, cider]])

    elif args.train_mode == 'case_study':
        if args.dataset_type in ['all', 'target'] :
            data_source = 'target'
        elif args.dataset_type == 'source':
            data_source = 'source'
        trg_lng = args['dataset'][data_source]['dataset_lng'][0]
        unilang_dataset = dataset[data_source][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        model_filename = args['common']['init_weights']
        Evaluator.case_study_eval(model, unilang_dataset['test'], dataset.token_dicts, model_filename=model_filename)

    # model: IModel, data_loader: DataLoader, token_dicts: TokenDicts,
    # model_filename = None,

    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
