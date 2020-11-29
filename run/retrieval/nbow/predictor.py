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
from ncc.utils.yaml import recursive_contractuser, recursive_expanduser


def main(model_path, input):
    state = load_checkpoint_to_cpu(model_path, arg_overrides={})
    args = state["args"]
    args = recursive_contractuser(args)
    args = recursive_expanduser(args)
    task = tasks.setup_task(args)  # load src/tgt dicts
    model = task.build_model(args)
    model.load_state_dict(state["model"])
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.device_count() - 1)
        model.cuda()
    if args['common']['fp16'] and use_cuda:
        model.half()
    model.eval()

    # load code_tokens dataset
    code_dataset = task.load_search_dataset(split=args['dataset']['gen_subset'])
    # construct similarities
    import numpy as np
    similarities = np.zeros(shape=len(code_dataset), dtype=np.float16)
    # query embeddding
    query_tokens = task.encode_query_input(input)
    if use_cuda:
        query_tokens = utils.move_to_cuda(query_tokens)
    query_tokens = model.tgt_encoder(query_tokens)
    # code embeddding
    for idx, code_tokens in enumerate(code_dataset):
        if use_cuda:
            code_tokens = utils.move_to_cuda(code_tokens)
        code_tokens = code_tokens.unsqueeze(dim=0)
        code_tokens = model.src_encoder(code_tokens)
        sim = (query_tokens * code_tokens).sum()
        similarities[idx] = sim.item()
    max_idx = np.argmax(similarities)
    with open(args['eval']['code_file'], 'r') as reader:
        for idx, line in enumerate(reader):
            if idx == max_idx:
                break
    return line


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Command Interface")

    parser.add_argument(
        "--model", "-m", type=str, help="pytorch model path",
        default=os.path.expanduser(
            "~/.ncc/code_search_net/retrieval/ruby/data-mmap/nbow/checkpoints/checkpoint_best.pt")
    )
    docstring = "Create a missing file if the path is valid."
    docstring = "Assign the value to the given attribute of the item"
    docstring = "Find adapter for the given request (resource, action, format)\n @raise [String]."
    docstring = "Validate the requested filter query strings. If all filters are valid\n then return them as {Hash hashes}, otherwise halt 400 Bad Request and\n return JSON error response."
    parser.add_argument(
        "--input", "-i", type=str, help="model input",
        # default=code_tokens
        default=docstring,
    )
    args = parser.parse_args()
    pred_sentence = main(args.model, args.input)
    print(pred_sentence)


if __name__ == '__main__':
    cli_main()
