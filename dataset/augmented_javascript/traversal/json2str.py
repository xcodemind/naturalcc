import os
import glob
import ujson
import argparse
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def get_file_line_num(f):
    f.seek(0)
    num = sum(1 for line in f)
    f.seek(0)
    return num


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


def json2str(in_file, out_file, start=0, end=0):
    with open(in_file, 'r') as reader, open(out_file, 'w') as writer:
        reader.seek(start)
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = ujson.loads(line)
            print(' '.join(line).replace('\n', '[CLS]'), file=writer)
            line = safe_readline(reader)


def merge_attr_files(file_patterns):
    """shell cat"""
    tgt_file = file_patterns[0]
    tgt_file = tgt_file[:str.rfind(tgt_file, '.')]

    print('Merge {}* to {}'.format(tgt_file, tgt_file))
    with open(tgt_file, 'w') as writer:
        for file in file_patterns:
            with open(file, 'r') as reader:
                shutil.copyfileobj(reader, writer)
            os.remove(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download code_search_net dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--in_files", "-i",
        default='~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/train.traversal',
        type=str, nargs='+', help="data directory of flatten attribute(load)",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    args.in_files = [os.path.expanduser(args.in_files)] if type(args.in_files) == str \
        else [os.path.expanduser(file) for file in args.in_files]
    # with Pool(args.cores) as mpool:
    #     for in_file in args.in_files:
    #         offsets = find_offsets(in_file, args.cores)
    #         result = [
    #             mpool.apply_async(
    #                 json2str,
    #                 (in_file, in_file + f'.str.{idx}', offsets[idx], offsets[idx + 1])
    #             )
    #             for idx in range(args.cores)
    #         ]
    #         result = [res.get() for res in result]
    #         merge_attr_files(file_patterns=[in_file + f'.str.{idx}' for idx in range(args.cores)])

    for in_file in args.in_files:
        with open(in_file, 'r') as reader, open(in_file + '.str', 'w') as writer:
            for line in tqdm(reader, total=get_file_line_num(reader)):
                line = ujson.loads(line)
                print(' '.join(line).replace('\n', '[SEP]'), file=writer)
