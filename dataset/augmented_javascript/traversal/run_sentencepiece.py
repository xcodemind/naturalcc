import os
import ujson
import argparse
from collections import Counter
from multiprocessing import Pool, cpu_count
import itertools
from ncc.utils.mp_ppool import PPool, cpu_count
import sentencepiece as spm


def find_offsets(filename, num_chunks):
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def spm_train(
    input: str, model_prefix: str, vocab_size: int, character_coverage=0.9995, model_type='unigram',
    const_tokens=None,
):
    if const_tokens:
        const_tokens_suffix = ' --user_defined_symbols={}'.format(','.join(['[SEP]'] + const_tokens))
    else:
        const_tokens_suffix = ''
    command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} " \
              f"--character_coverage={character_coverage} --model_type={model_type} --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3" \
              f" --unk_piece=[UNK] --pad_piece=[PAD] --hard_vocab_limit=false" + const_tokens_suffix
    print(command)
    spm.SentencePieceTrainer.Train(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--in_files", "-i",
        default='~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/train.traversal.str',
        type=str, nargs='+', help="data directory of flatten attribute(load)",
    )
    parser.add_argument(
        "--bpe_model", "-m",
        default='~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/.traversal.str',
        type=str, help="out file to save tree node type names",
    )
    parser.add_argument(
        "--const_tokens", "-t",
        default='~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/.ast.node_types',
        type=str, help="out file to save tree node type names",
    )
    parser.add_argument("--vocab_size", type=int, default=8000, help='token dictionary size')
    args = parser.parse_args()
    args.in_files = [os.path.expanduser(args.in_files)] if type(args.in_files) == str \
        else [os.path.expanduser(file) for file in args.in_files]
    args.bpe_model = os.path.expanduser(args.bpe_model)
    args.const_tokens = os.path.expanduser(args.const_tokens)

    with open(args.const_tokens, 'r') as reader:
        const_tokens = [ujson.loads(line)[0] for line in reader]
    assert args.vocab_size > len(const_tokens), (args.vocab_size, len(const_tokens))

    spm_train(','.join(args.in_files), model_prefix=args.bpe_model, vocab_size=args.vocab_size,
              const_tokens=const_tokens)
