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
from ncc import (tasks, LOGGER)
from ncc.utils import utils
from ncc.utils.checkpoint_utils import load_checkpoint_to_cpu

import ujson


def load_state(model_path):
    state = load_checkpoint_to_cpu(model_path, arg_overrides={})
    args = state["args"]
    task = tasks.setup_task(args)  # load src/tgt dicts
    model = task.build_model(args)
    model.load_state_dict(state["model"])
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if args['common']['fp16'] and use_cuda:
        model.half()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.device_count() - 1)
        model.cuda()
    model.eval()
    del state
    return args, task, model, use_cuda


def main(model_path, input):
    args, task, model, use_cuda = load_state(model_path)
    generator = task.build_generator(args)

    with open(os.path.join(os.path.dirname(__file__), 'config/case.txt'), 'r') as reader:
        for input in reader:
            # encode input (and feed into gpu)
            [raw_input, ref] = ujson.loads(input)

            input = task.encode_input(raw_input)
            if use_cuda:
                input = utils.move_to_cuda(input)

            # feed input into model
            output = generator.generate(models=[model], sample=input)
            # decode
            output = task.decode_output(output)
            if ' '.join(ref) == output:
                LOGGER.info(ujson.dumps([raw_input, output]))
    return output


def cli_main():
    """
    nohup python -m run.summarization.neural_transformer.predictor > run/summarization/neural_transformer/case.log 2>&1 &
    """

    # code summarization
    model_path = os.path.expanduser("~/.ncc/demo/summarization/neural_transformer/python_wan.pt")

    import argparse
    parser = argparse.ArgumentParser(description="Command Interface")
    parser.add_argument("--model", "-m", type=str, help="pytorch model path", default=model_path)
    parser.add_argument("--input", "-i", type=str, help="model input")
    args = parser.parse_args()
    model_output = main(args.model, args.input)


if __name__ == '__main__':
    cli_main()
