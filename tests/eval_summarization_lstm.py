# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import numpy as np
from collections import namedtuple
from ncc import LOGGER
from ncc.utils.util_file import load_yaml
from ncc.tasks.summarization import load_langpair_dataset
from ncc.tasks import NccTask
from ncc import tasks
import torch
import torch.nn as nn
from ncc.data.summarization.language_pair_dataset import collate
from ncc.data import iterators
from ncc.logging import metrics, progress_bar
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.trainer.ncc_trainer import Trainer
from ncc.utils.file_utils import remove_files
from ncc.utils import utils
from tqdm import tqdm
import torch.optim as optim
from third_party.pycocoevalcap.bleu import corpus_bleu
from third_party.pycocoevalcap.rouge import Rouge
from third_party.pycocoevalcap.meteor import Meteor
from collections import OrderedDict, Counter
import json

def eval_accuracies(hypotheses, references, sources=None,
                    filename=None, mode='dev'):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, ind_bleu = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    # f1 = AverageMeter()
    # precision = AverageMeter()
    # recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        # _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
        #                                       references[key])
        # precision.update(_prec)
        # recall.update(_rec)
        # f1.update(_f1)
        if fw:
            # if copy_info is not None and print_copy_info:
            #     prediction = hypotheses[key][0].split()
            #     pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
            #               for j, word in enumerate(prediction)]
            #     pred_i = [' '.join(pred_i)]
            # else:
            pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            # logobj['references'] = references[key][0] if args.print_one_target \
            #     else references[key]
            logobj['references'] = references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            fw.write(json.dumps(logobj) + '\n')

    if fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100 #, precision.avg * 100, recall.avg * 100, f1.avg * 100


if __name__ == '__main__':
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('python-wan.yml')  # train_sl
    LOGGER.info(args_)
    print('args: ', type(args_))
    # yaml_file = os.path.join('../../../naturalcodev3/run/summarization/lstm2lstm/', 'config', args_.yaml)
    yaml_file = os.path.join('../../../naturalcodev3/run/summarization/lstm2lstm/', 'config', args_.yaml)
    yaml_file = os.path.realpath(yaml_file)
    # yaml_file = os.path.join('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/summarization/seq2seq/', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    # 0. Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])
    np.random.seed(args['common']['seed'])
    torch.manual_seed(args['common']['seed'])

    # Print args
    LOGGER.info(args)

    task = tasks.setup_task(args)  # task.tokenizer
    model = task.build_model(args)  # , config
    # parameters = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(parameters, 0.002, weight_decay=0)
    model.load_state_dict(torch.load(os.path.join(args['checkpoint']['save_dir'], 'e{}.pt'.format(0))))

    criterion = task.build_criterion(args)
    device = torch.device('cuda')
    # criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    model = model.to(device)
    trainer = Trainer(args, task, model, criterion)

    # Data
    data_path = os.path.expanduser('~/.ncc/python_wan/summarization/data-raw/python')
    # src_dict = NccTask.load_dictionary(args['dataset']['srcdict'])
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    src, tgt = args['task']['source_lang'], args['task']['target_lang']
    combine = False

    # evaluate
    """Run one full official validation. Uses exact spans and same
        exact match/F1 score computation as in the SQuAD script.
        Extra arguments:
            offsets: The character start/end indices for the tokens in each context.
            texts: Map of qid --> raw text of examples context (matches offsets).
            answers: Map of qid --> list of accepted answers.
    """
    # eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()

    valid_dataset = load_langpair_dataset(
        data_path, 'valid', src, src_dict, tgt, tgt_dict,
        combine=combine, dataset_impl=args['dataset']['dataset_impl'],
        upsample_primary=args['task']['upsample_primary'],
        left_pad_source=args['task']['left_pad_source'],
        left_pad_target=args['task']['left_pad_target'],
        max_source_positions=args['task']['max_source_positions'],
        max_target_positions=args['task']['max_target_positions'],
        load_alignments=args['task']['load_alignments'],
        truncate_source=args['task']['truncate_source'],
        append_eos_to_target=args['task']['append_eos_to_target'],
    )

    dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collater,
        # batch_sampler=batches[offset:],
        num_workers=args['dataset']['num_workers'],
        batch_size=args['dataset']['max_sentences']
    )
    pbar = tqdm(dataloader)
    total_loss = []
    count = 0
    # for epoch in range(200):
    # bleus = []
    hyps_, refs_, ids_ = [], [], []
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    for idx, sample in enumerate(pbar):
        # loss, sample_size, logging_output = trainer.valid_step(sample)
        hyps, refs, ids = trainer.valid_step(sample)
        hyps_.extend(hyps)
        refs_.extend(refs)
        ids_.extend(ids)
        # bleus.append(bleu.score)
    # for i in range(len(ids)):
    for key, pred, tgt in zip(ids_, hyps_, refs_):
        hypotheses[key] = [pred]
        references[key] = tgt if isinstance(tgt, list) else [tgt]
    bleu, rouge_l, meteor = eval_accuracies(hypotheses,
                                                                   references,
                                                                   filename='pred.txt')
    LOGGER.info('test valid official: '
                'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                (bleu, rouge_l, meteor))
    # print("average bleu: ", np.mean(bleus))
    exit()
    valid_dataset = load_langpair_dataset(
        data_path, 'valid', src, src_dict, tgt, tgt_dict,
        combine=combine, dataset_impl=args['dataset']['dataset_impl'],
        upsample_primary=args['task']['upsample_primary'],
        left_pad_source=args['task']['left_pad_source'],
        left_pad_target=args['task']['left_pad_target'],
        max_source_positions=args['task']['max_source_positions'],
        max_target_positions=args['task']['max_target_positions'],
        load_alignments=args['task']['load_alignments'],
        truncate_source=args['task']['truncate_source'],
        append_eos_to_target=args['task']['append_eos_to_target'],
    )

    # itr_val = task.get_batch_iterator(
    #     dataset=valid_dataset,
    #     max_tokens=args['dataset']['max_tokens'],
    #     max_sentences=args['dataset']['max_sentences'],
    #     max_positions=utils.resolve_max_positions(
    #         task.max_positions(),
    #         trainer.get_model().max_positions(),
    #     ),
    #     ignore_invalid_inputs=True,
    #     required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
    #     seed=args['common']['seed'],
    #     num_shards=1,
    #     shard_id=0,
    #     num_workers=0,  # args['dataset']['num_workers'],
    #     # epoch=0,
    # ).next_epoch_itr(shuffle=False)
    itr_val = task.get_batch_iterator(
        dataset=valid_dataset,
        max_tokens=args['dataset']['max_tokens_valid'],
        max_sentences=args['dataset']['max_sentences_valid'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
        seed=args['common']['seed'],
        num_shards=args['distributed_training']['distributed_world_size'],
        shard_id=args['distributed_training']['distributed_rank'],
        num_workers=args['dataset']['num_workers'],
    ).next_epoch_itr(shuffle=False)

    with torch.no_grad():
        # itr = epoch_itr_val.next_epoch_itr(
        #     fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
        #     shuffle=(epoch_itr_val.next_epoch_idx > args['dataset']['curriculum']),
        # )

        progress_val = progress_bar.progress_bar(
            itr_val,
            log_format=args['common']['log_format'],
            log_interval=args['common']['log_interval'],
            epoch=epoch_itr.epoch,
            prefix=f"valid on  subset",
            tensorboard_logdir=(
                args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
        )

        # task specific setup per epoch
        # task.begin_epoch(epoch_itr.epoch, trainer.get_model())
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress_val:
                print('sample: ', sample)
                trainer.valid_step(sample)
                exit()
                # for idx, sample in enumerate(samples):
                    # batch_size = ex['batch_size']
                    # ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
                    # ex_ids = sample['id']
                    # print('ex_ids: ', ex_ids)
                    # predictions, targets, copy_info = model.predict(sample, replace_unk=True)
                    #
                    # src_sequences = [code for code in sample['code_text']]
                    # # examples += batch_size
                    # for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                    #     hypotheses[key] = [pred]
                    #     references[key] = tgt if isinstance(tgt, list) else [tgt]
                    #     sources[key] = src
                # if copy_info is not None:
                #     copy_info = copy_info.cpu().numpy().astype(int).tolist()
                #     for key, cp in zip(ex_ids, copy_info):
                #         copy_dict[key] = cp

                # pbar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

        # copy_dict = None if len(copy_dict) == 0 else copy_dict
        bleu, rouge_l, meteor = eval_accuracies(hypotheses, references,
                                                                       sources=sources,
                                                                       filename='pred.txt',#args.pred_file,
                                                                       # print_copy_info=args.print_copy_info,
                                                                    )
        result = dict()
        result['bleu'] = bleu
        result['rouge_l'] = rouge_l
        result['meteor'] = meteor
        # result['precision'] = precision
        # result['recall'] = recall
        # result['f1'] = f1

        # if mode == 'test':
        #     logger.info('test valid official: '
        #                 'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
        #                 (bleu, rouge_l, meteor) +
        #                 'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
        #                 'examples = %d | ' %
        #                 (precision, recall, f1, examples) +
        #                 'test time = xx (s)')
        #
        # else:
        #     logger.info('dev valid official: Epoch = %d | ' %
        #                 (global_stats['epoch']) +
        #                 'bleu = %.2f | rouge_l = %.2f | '
        #                 'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
        #                 (bleu, rouge_l, precision, recall, f1, examples) +
        #                 'valid time = xx (s)')
