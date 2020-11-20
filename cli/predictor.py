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
    # encode input (and feed into gpu)
    input = task.encode_input(input)
    if use_cuda:
        input = utils.move_to_cuda(input)
    # feed input into model
    generator = task.build_generator(args)
    # from ipdb import set_trace
    # set_trace()
    output = generator.generate(models=[model], sample=input)
    # decode
    output = task.decode_output(output)
    return output


def cli_main():
    # code summarization
    model_path = os.path.expanduser("~/.ncc/demo/summarization/seq2seq/python_wan.pt")
    model_path = os.path.expanduser("~/.ncc/demo/summarization/neural_transformer/python_wan.pt")
    codes = [
        "def positional(max_positional_args):\n\tdef positional_decorator(wrapped):\n\t\t@functools.wraps(wrapped)\n\t\tdef positional_wrapper(*args, **kwargs):\n\t\t\tif (len(args) > max_posi      tional_args):\n\t\t\t\tplural_s = ''\n\t\t\t\tif (max_positional_args != 1):\n\t\t\t\t\tplural_s = 's'\n\t\t\t\tmessage = ('%s()\ttakes\tat\tmost\t%d\tpositional\targument%s\t(%d\tgive      n)' % (wrapped.__name__, max_positional_args, plural_s, len(args)))\n\t\t\t\tif (positional_parameters_enforcement == POSITIONAL_EXCEPTION):\n\t\t\t\t\traise TypeError(message)\n\t\t\t      \telif (positional_parameters_enforcement == POSITIONAL_WARNING):\n\t\t\t\t\tlogger.warning(message)\n\t\t\t\telse:\n\t\t\t\t\tpass\n\t\t\treturn wrapped(*args, **kwargs)\n\t\treturn p      ositional_wrapper\n\tif isinstance(max_positional_args, six.integer_types):\n\t\treturn positional_decorator\n\telse:\n\t\t(args, _, _, defaults) = inspect.getargspec(max_positional_ar      gs)\n\t\treturn positional((len(args) - len(defaults)))(max_positional_args)\n",
        "def getCarveIntersectionFromEdge(edge, vertexes, z):\n\tfirstVertex = vertexes[edge.vertexIndexes[0]]\n\tfirstVertexComplex = firstVertex.dropAxis(2)\n\tsecondVertex = vertexes[edge.v      ertexIndexes[1]]\n\tsecondVertexComplex = secondVertex.dropAxis(2)\n\tzMinusFirst = (z - firstVertex.z)\n\tup = (secondVertex.z - firstVertex.z)\n\treturn (((zMinusFirst * (secondVerte      xComplex - firstVertexComplex)) \/ up) + firstVertexComplex)\n",
        "def MessageEncoder(field_number, is_repeated, is_packed):\n\ttag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)\n\tlocal_EncodeVarint = _EncodeVarint\n\tassert (not i      s_packed)\n\tif is_repeated:\n\t\tdef EncodeRepeatedField(write, value):\n\t\t\tfor element in value:\n\t\t\t\twrite(tag)\n\t\t\t\tlocal_EncodeVarint(write, element.ByteSize())\n\t\t\t      \telement._InternalSerialize(write)\n\t\treturn EncodeRepeatedField\n\telse:\n\t\tdef EncodeField(write, value):\n\t\t\twrite(tag)\n\t\t\tlocal_EncodeVarint(write, value.ByteSize())\n\      t\t\treturn value._InternalSerialize(write)\n\t\treturn EncodeField\n",
    ]
    gts = [
        "a decorator to declare that only the first n arguments my be positional.",
        "get the complex where the carve intersects the edge .",
        "returns an encoder for a message field .",
    ]

    import argparse
    parser = argparse.ArgumentParser(description="Command Interface")
    parser.add_argument("--model", "-m", type=str, help="pytorch model path", default=model_path)
    parser.add_argument("--input", "-i", type=str, help="model input")
    args = parser.parse_args()
    for code, gt in zip(codes, gts):
        args.input = code
        model_output = main(args.model, args.input)
        # print(f"==> code <==\n{code}\n==> gt <==\n{model_output}")
        print(f"==> gt <== {gt}")
        print(f"==> pr <== {model_output}")
        # exit()


if __name__ == '__main__':
    cli_main()
