#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import re
import torch
from ncc import tasks
from ncc.utils import utils
from ncc.utils.checkpoint_utils import load_checkpoint_to_cpu
from ncc import LOGGER


def main(model_path, input):
    LOGGER.info('Load model from {}'.format(model_path))
    state = load_checkpoint_to_cpu(model_path, arg_overrides={})
    args = state["args"]
    task = tasks.setup_task(args)  # load src/tgt dicts
    model = task.build_model(args)
    model.load_state_dict(state["model"])
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    use_cuda = 0
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.device_count() - 1)
        model.cuda()
    model.eval()
    if args['common']['fp16'] and use_cuda:
        model.half()

    # TODO: source tensor should be handled in corresponding task scripts. here we only use seq2seq pipeline for instance.
    # from ipdb import set_trace
    # set_trace()
    intput_ids = task.target_dictionary.encode_line1(input, line_tokenizer=None, add_if_not_exist=False)
    src_input_ids = intput_ids.long().unsqueeze(dim=0)

    sample = {
        'net_input': {
            'src_tokens': src_input_ids[:, :-1],
        },
    }
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    generator = task.sequence_completor
    net_output = generator.complete(models=[model], sample=sample)

    # from ipdb import set_trace
    # set_trace()

    pred_prob = torch.softmax(net_output[0][0, -1, :], dim=-1)
    topk_prob, topk_idx = pred_prob.topk(k=10, dim=-1)
    # remove unk/eos/bos/pad
    topk_info = [(round(prob.item(), 6), idx.item()) for prob, idx in zip(topk_prob, topk_idx)][:5]
    topk_info = [(task.target_dictionary[idx], prob) for prob, idx in topk_info]
    pred_sentence = [
        (input[:-1] + [topk_token], topk_prob)
        for topk_token, topk_prob in topk_info
    ]
    return topk_info, pred_sentence


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Command Interface")

    parser.add_argument(
        "--model", "-m", type=str, help="pytorch model path",
        # default=os.path.expanduser("~/.ncc/py150/completion/data-mmap/seqrnn/checkpoints/checkpoint_last.pt")
        default=os.path.expanduser("~/.ncc/py150/completion/data-mmap/seqrnn/checkpoints/checkpoint_best.pt")
    )

    # code_tokens = ['def', 'addition', 'arg1', 'arg2', 'sum', '=', 'arg1', '+', 'arg2', 'return', 'sum']
    code_tokens = [
        "Django's command line utility.",
        'os',
        'sys',
        '__name__',
        '__main__',
        'os',
        'environ',
        'setdefault',
        'DJANGO_SETTINGS_MODULE',
        # 'project.settings',
        # 'django.core.management',
        # 'execute_from_command_line',
        # 'execute_from_command_line',
        # 'sys',
        # 'argv',
    ]

    parser.add_argument(
        "--input", "-i", type=str, help="model input",
        # default=code_tokens
        default=code_tokens,
    )
    args = parser.parse_args()

    out = main(args.model, args.input)
    print(out[0])


if __name__ == '__main__':
    cli_main()
