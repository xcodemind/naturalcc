#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import sys
from collections import namedtuple
import math
import torch
from ncc import LOGGER
from ncc import tasks
from ncc.eval import bleu_scorer
from ncc.utils import checkpoint_utils
from ncc.logging import progress_bar
from ncc.utils import utils
from ncc.utils.util_file import load_yaml
from ncc.logging.meters import StopwatchMeter, TimeMeter

import torch.nn.functional as F


def sample_beam(args, sample, tgt_dict, model):
    batch_size = len(sample['id'])
    DEVICE = sample['target'].device
    input = torch.LongTensor([batch_size, 1]).fill_(tgt_dict.eos()).to(DEVICE)

    # print('sample_beam')
    beam_size = args['eval']['beam']
    seq_length = 20
    seq_all = input.new_zeros(batch_size, seq_length, beam_size).long()
    seq = input.new_zeros(batch_size, seq_length).long()
    seqLogprobs = input.new_zeros(batch_size, seq_length).long()

    hidden_state = model.encoder(**sample['net_input'])['encoder_out'][1:]
    hidden_state=[]

    hidden_size = hidden_state.size(-1)
    # lets process every image independently for now, for simplicity
    done_beams = [[] for _ in range(batch_size)]
    # print('done_beams: ', done_beams)
    for k in range(batch_size):
        # copy the hidden state for beam_size time.
        state = []
        for state_tmp in hidden_state:
            state.append(state_tmp[:, k, :].reshape(1, 1, -1).expand(1, beam_size, hidden_size).clone())
        # print('state: ')
        # print(state)
        state = tuple(state)
        beam_seq = input.new_zeros(seq_length, beam_size).long()
        # print('beam_seq: ', beam_seq.type(), beam_seq.size())
        # print(beam_seq)
        beam_seq_logprobs = input.new_zeros(seq_length, beam_size).long()
        # print('beam_seq_logprobs: ', beam_seq_logprobs.type(), beam_seq_logprobs.size())
        # print(beam_seq_logprobs)
        beam_logprobs_sum = input.new_zeros(beam_size)  # running sum of logprobs for each beam
        # print('beam_logprobs_sum: ', beam_logprobs_sum.type(), beam_logprobs_sum.size())
        for t in range(seq_length + 1):
            # print('step-t: ', t)
            if t == 0:  # input <bos>
                it = input.resize_(1, beam_size).fill_(BOS)  # .data
                xt = self.wemb(it.detach())
            else:
                """perform a beam merge. that is,
                for every previous beam we now many new possibilities to branch out
                we need to resort our beams to maintain the loop invariant of keeping
                the top beam_size most likely sequences."""
                logprobsf = logprobs.float().cpu()  # lets go to CPU for more efficiency in indexing operations
                ys, ix = torch.sort(logprobsf, 1,
                                    True)  # sorted array of logprobs along each previous beam (last true = descending)
                candidates = []
                cols = min(beam_size, ys.size(1))
                rows = beam_size
                if t == 1:  # at first time step only the first beam is active
                    rows = 1
                for cc in range(cols):  # for each column (word, essentially)
                    for qq in range(rows):  # for each beam expansion
                        # compute logprob of expanding beam q with word in (sorted) position c
                        local_logprob = ys[qq, cc]
                        if beam_seq[t - 2, qq] == self.embed_size:  # self.opt.ninp:
                            local_logprob.data.fill_(-9999)
                        # print('local_logprob: ', local_logprob.type(), local_logprob.size())
                        # print(local_logprob)
                        # print('beam_logprobs_sum[qq]: ', beam_logprobs_sum[qq].type(), beam_logprobs_sum[qq].size())
                        # print(beam_logprobs_sum[qq])
                        candidate_logprob = beam_logprobs_sum[qq] + local_logprob
                        candidates.append({'c': ix.data[qq, cc], 'q': qq, 'p': candidate_logprob.item(),
                                           'r': local_logprob.item()})

                candidates = sorted(candidates, key=lambda x: -x['p'])
                # print('candidates: ', candidates)
                # construct new beams
                new_state = [_.clone() for _ in state]
                # print('new_state: ')
                # print(new_state)
                if t > 1:
                    # well need these as reference when we fork beams around
                    beam_seq_prev = beam_seq[:t - 1].clone()
                    beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1].clone()
                for vix in range(beam_size):
                    v = candidates[vix]
                    # fork beam index q into index vix
                    if t > 1:
                        beam_seq[:t - 1, vix] = beam_seq_prev[:, v['q']]
                        beam_seq_logprobs[:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                    # rearrange recurrent states
                    for state_ix in range(len(new_state)):
                        # copy over state in previous beam q to new beam at vix
                        new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                    # append new end terminal at the end of this beam
                    beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                    beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                    beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                    if v['c'] == self.opt.ninp or t == seq_length:
                        # END token special case here, or we reached the end.
                        # add the beam to a set of done beams
                        done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                              'logps': beam_seq_logprobs[:, vix].clone(),
                                              'p': beam_logprobs_sum[vix]
                                              })

                # encode as vectors
                it = beam_seq[t - 1].reshape(1, -1)
                xt = self.wemb(it.cuda())

            if t >= 1:
                state = new_state

            output, state = self.rnn(xt, hidden=state)

            output = F.dropout(output, self.dropout, training=self.training)
            decoded = self.generate_linear(output.reshape(output.size(0) * output.size(1), output.size(2)))
            logprobs = F.log_softmax(decoded, dim=1)  # self.beta *

        done_beams[k] = sorted(done_beams[k], key=lambda x: -x['p'])
        seq[:, k] = done_beams[k][0]['seq']  # the first beam has highest cumulative score
        seqLogprobs[:, k] = done_beams[k][0]['logps']
        for ii in range(beam_size):
            seq_all[:, k, ii] = done_beams[k][ii]['seq']

    # return the samples and their log likelihoods
    return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)


def main(args):
    assert args['eval']['path'] is not None, '--path required for generation!'
    assert not args['eval']['sampling'] or args['eval']['nbest'] == args['eval']['beam'], \
        '--sampling requires --nbest to be equal to --beam'
    assert args['eval']['replace_unk'] is None or args['dataset']['dataset_impl'] == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    if args['eval']['results_path'] is not None:
        os.makedirs(args['eval']['results_path'], exist_ok=True)
        output_path = os.path.join(args['eval']['results_path'], 'generate-{}.txt'.format(args['eval']['gen_subset']))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def _main(args, output_file):
    if args['dataset']['max_tokens'] is None and args['dataset']['max_sentences'] is None:
        args['dataset']['max_tokens'] = 12000
    LOGGER.info(args)

    use_cuda = torch.cuda.is_available() and not args['common']['cpu']

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['gen_subset'])

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args['eval']['no_beamable_mm'] else args['eval']['beam'],
            need_attn=args['eval']['print_alignment'],
        )
        if _model_args['common']['fp16']:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args['eval']['replace_unk'])

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args['dataset']['gen_subset']),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=_model_args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=_model_args['dataset']['required_batch_size_multiple'],
        # num_shards=_model_args['dataset']['num_shards'],
        # shard_id=_model_args['dataset']['shard_id'],
        num_workers=_model_args['dataset']['num_workers'],
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=_model_args['common']['log_format'],
        log_interval=_model_args['common']['log_interval'],
        default_log_format=('tqdm' if not _model_args['common']['no_progress_bar'] else 'none'),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    from ncc.eval import search
    search.BeamSearch(tgt_dict)

    # Generate and compute BLEU score
    if args['eval']['sacrebleu']:
        scorer = bleu_scorer.SacrebleuScorer()
    else:
        scorer = bleu_scorer.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()

    model = models[0]
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        prefix_tokens = None
        if args['eval']['prefix_size'] > 0:
            prefix_tokens = sample['target'][:, :args['eval']['prefix_size']]

        sample_beam(args, sample, tgt_dict, model)

        gen_timer.start()
        hypos = task.inference_step(generator, models, sample, prefix_tokens)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args['dataset']['gen_subset']).src.get_original_text(sample_id)
                target_str = task.dataset(args['dataset']['gen_subset']).tgt.get_original_text(sample_id)
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args['eval']['remove_bpe'])
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(target_tokens, args['eval']['remove_bpe'], escape_unk=True)

            if not args['eval']['quiet']:
                if src_dict is not None:
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args['eval']['nbest']]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args['eval']['remove_bpe'],
                )

                if not args['eval']['quiet']:
                    score = hypo['score'] / math.log(2)  # convert to base 2
                    print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)

                    if args['eval']['print_alignment']:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        ), file=output_file)

                    if args['eval']['print_step']:
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                    # if getattr(args, 'retain_iter_history', False):
                    if args['eval']['retain_iter_history']:
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or args['eval']['remove_bpe'] is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(target_str, hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample['nsentences']

        break

    LOGGER.info('NOTE: hypothesis and token scores are output in base 2')
    LOGGER.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        LOGGER.info('Generate {} with beam={}: {}'.format(args['dataset']['gen_subset'], args['eval']['beam'],
                                                          scorer.result_string()))

    return scorer


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('ruby.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
