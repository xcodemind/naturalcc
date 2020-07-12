# -*- coding: utf-8 -*-

import os
import json
import itertools
import numpy as np

from multiprocessing import Pool, cpu_count
from collections import (
    Counter,
    namedtuple,
)

from ncc import (
    tasks,
    LOGGER,
)
from ncc.utils.util_file import load_yaml
from ncc.data import indexed_dataset
from ncc.data.dictionary import Dictionary
from ncc.data.tools.binarizer import Binarizer
from ncc.utils.mp_ppool import PPool

from dataset.utils.ast import tranv_trans


def file_name(prefix, lang):
    """get file name without diretory"""
    fname = prefix if lang is None else prefix + ".{lang}".format(lang=lang)
    return fname


def dest_path(dest_dir, prefix, lang):
    """get file path"""
    return os.path.join(dest_dir, file_name(prefix, lang))


def train_path(trainpref, lang):
    """get train data file name"""
    return "{}{}".format(trainpref, ("." + lang) if lang else "")


def tokenize_func(line):
    return tranv_trans.get_dfs(json.loads(line))


def dataset_dest_prefix(args, output_prefix, lang):
    base = os.path.join(args['preprocess']['destdir'], output_prefix)
    return "{}.{}".format(base, lang)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def binarize(args, filename, vocab, output_prefix, lang, make_dataset_fn, offset, end):
    """binarize function for multi-processing"""
    src_vocab, mask_dict = vocab

    ds = {
        'data': indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, lang, "mmap"),
            impl=args['preprocess']['dataset_impl'], vocab_size=len(src_vocab)  # np.int32
        ),
        'ext': indexed_dataset.MMapIndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, 'ext', "mmap"), dtype=np.uint16
        ),
    }
    if args['preprocess']['id_type']:
        ds['ids'] = {
            cls: indexed_dataset.make_builder(
                dataset_dest_file(args, output_prefix, cls, "mmap"),
                impl=args['preprocess']['dataset_impl'],  # np.int32
            )
            for cls in tranv_trans.IDS_CLS
        }
    if args['preprocess']['rel_mask']:
        ds['mask'] = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, 'mask', "mmap"),
            impl=args['preprocess']['dataset_impl'], vocab_size=len(mask_dict)  # np.uint16
        )

    def consumer(data, ext, ids, mask):
        ds['data'].add_item(data)
        ds['ext'].add_item(ext)
        for key, value in ids.items():
            ds['ids'][key].add_item(value)
        ds['mask'].add_item(mask)

    res = Binarizer.binarize_trav_trans(
        filename, vocab, consumer,
        tokenize=make_dataset_fn,
        offset=offset, end=end
    )
    ds['data'].finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    ds['ext'].finalize(dataset_dest_file(args, output_prefix, 'ext', "idx"))
    for cls in tranv_trans.IDS_CLS:
        ds['ids'][cls].finalize(dataset_dest_file(args, output_prefix, cls, "idx"))
    ds['mask'].finalize(dataset_dest_file(args, output_prefix, 'mask', "idx"))
    return res


MAX_BATCH_SIZE = 2


def main(args):
    LOGGER.info('{} task'.format(args['preprocess']['task']))
    # mkdir for dest directory
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    # 1. ***************build vocabulary***************
    task = tasks.get_task(args['preprocess']['task'])

    def get_dicts():
        # token dict
        if args['preprocess']['srcdict']:
            LOGGER.info('Loading dict from {}'.format(args['preprocess']['srcdict']))
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"
            dict_filename = dest_path(args['preprocess']['destdir'], "dict",
                                      args['preprocess']['source_lang']) + ".json"
            if os.path.exists(dict_filename):
                LOGGER.info('Loading dict from {}'.format(dict_filename))
                src_dict = task.load_dictionary(dict_filename)
            else:
                LOGGER.info('Building dict {}'.format(dict_filename))
                src_dict = task.build_dictionary(
                    [train_path(args['preprocess']['trainpref'], args['preprocess']['source_lang'])],
                    tokenize_func=tokenize_func,
                    workers=args['preprocess']['workers'],
                    threshold=args['preprocess']['thresholdsrc'],
                    nwords=args['preprocess']['nwordssrc'],
                    padding_factor=args['preprocess']['padding_factor'],
                )
                #  save dictionary
                LOGGER.info('Saving dict from {}'.format(dict_filename))
                src_dict.save_json(dict_filename)

        # mask dict
        if args['preprocess']['mask_dict']:
            mask_dict = task.load_dictionary(args['preprocess']['mask_dict'])
        else:
            # mask dict token
            # mask dict tokens: i|j -> relative mask, i>0, j>=0. i,j are a small number. Here we assign them as 20
            mask_dict_filename = dest_path(args['preprocess']['destdir'], "dict", 'rel_mask') + ".json"
            if os.path.exists(mask_dict_filename):
                LOGGER.info('Loading mask dict from {}'.format(mask_dict_filename))
                mask_dict = task.load_dictionary(mask_dict_filename)
            else:
                LOGGER.info('Building mask dict {}'.format(mask_dict_filename))
                _MAX_LEVEL = 20
                mask_dict = Dictionary()
                for i, j in itertools.product(range(1, _MAX_LEVEL), range(_MAX_LEVEL)):
                    mask_dict.add_symbol('{}|{}'.format(i, j))
                #  save dictionary
                LOGGER.info('Saving mask dict from {}'.format(mask_dict_filename))
                mask_dict.save_json(mask_dict_filename)
        return src_dict, mask_dict

    src_dict, mask_dict = get_dicts()

    # 2. ***************build dataset********************
    def ast2ids(ast):
        ids = {}
        if args['preprocess']['id_type'] in {"leaf", "all"}:
            ids.update(tranv_trans.get_leaf_ids(ast))
        if args['preprocess']['id_type'] in {"value", "all"}:
            ids.update(tranv_trans.get_value_ids(ast))
        if args['preprocess']['id_type'] in {"type", "all"}:
            ids.update(tranv_trans.get_type_ids(ast))
        return ids

    def make_raw_dataset(line, ids=False, rel_mask=False):
        """
        because only 1 thread is allowed to write file, we have to use multi-processing for deal with data
        and merge results from CPUs into a block and then dumps such block.
        """
        line = json.loads(line.strip())
        sep_asts = tranv_trans.separate_dps(line, args['preprocess']['n_ctx'])
        if rel_mask:
            masks = tranv_trans.get_rel_masks(line, max_len=args['preprocess']['n_ctx'])
            masks = tranv_trans.separate_rel_mask(masks, max_len=args['preprocess']['n_ctx'])
        else:
            masks = [None] * len(sep_asts)

        aug_dps = []
        for idx, (ast, ext) in enumerate(sep_asts):
            if len(ast) > 1:
                ids = ast2ids(ast) if ids else None
                aug_dps.append([tranv_trans.get_dfs(ast), ext, ids, list(itertools.chain(*masks[idx]))])
        return aug_dps

    def make_mmap_dataset(src_vocab, mask_vocab, input_prefix, output_prefix, lang, num_workers):
        LOGGER.info("[{}] Dictionary: {} types, Mask Dictionary: {} types". \
                    format(lang, len(src_vocab) - 1, len(mask_vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        _make_dataset_func = lambda line: make_raw_dataset(
            line, ids=bool(args['preprocess']['id_type']), rel_mask=bool(args['preprocess']['rel_mask'])
        )

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        lang = args['preprocess']['source_lang']
        offsets = Binarizer.find_offsets(input_file, num_workers)

        if num_workers > 1:
            with PPool(num_workers - 1) as pool:
                params = []
                for worker_id in range(1, num_workers):
                    params.append((
                        args,
                        input_file,
                        (src_vocab, mask_vocab),
                        "{}{}".format(output_prefix, worker_id),
                        lang,
                        _make_dataset_func,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ))
                result = pool.feed(func=binarize, params=params)
                for res in result:
                    merge_result(res)

        def consumer(data, ext, ids, mask):
            ds['data'].add_item(data)
            ds['ext'].add_item(ext)
            for key, value in ids.items():
                ds['ids'][key].add_item(value)
            ds['mask'].add_item(mask)

        ds = {
            'data': indexed_dataset.make_builder(
                dataset_dest_file(args, output_prefix, lang, "mmap"),
                impl=args['preprocess']['dataset_impl'], vocab_size=len(src_vocab)  # np.int32
            ),
            'ext': indexed_dataset.MMapIndexedDatasetBuilder(
                dataset_dest_file(args, output_prefix, 'ext', "mmap"), dtype=np.uint16
            ),
        }
        if args['preprocess']['id_type']:
            ds['ids'] = {
                cls: indexed_dataset.make_builder(
                    dataset_dest_file(args, output_prefix, cls, "mmap"),
                    impl=args['preprocess']['dataset_impl'],  # np.int32
                )
                for cls in tranv_trans.IDS_CLS
            }
        if args['preprocess']['rel_mask']:
            ds['mask'] = indexed_dataset.make_builder(
                dataset_dest_file(args, output_prefix, 'mask', "mmap"),
                impl=args['preprocess']['dataset_impl'], vocab_size=len(mask_dict)  # np.uint16
            )

        merge_result(
            Binarizer.binarize_trav_trans(
                input_file, (src_vocab, mask_vocab), consumer,
                tokenize=_make_dataset_func,
                offset=0, end=offsets[1]
            )
        )

        if num_workers > 1:
            def remove_mmap_idx(temp_file_path):
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                # data
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds['data'].merge_file_(temp_file_path)
                remove_mmap_idx(temp_file_path)
                # ext
                temp_file_path = dataset_dest_prefix(args, prefix, 'ext')
                ds['ext'].merge_file_(temp_file_path)
                remove_mmap_idx(temp_file_path)
                # ids
                for cls in tranv_trans.IDS_CLS:
                    temp_file_path = dataset_dest_prefix(args, prefix, cls)
                    ds['ids'][cls].merge_file_(temp_file_path)
                    remove_mmap_idx(temp_file_path)
                # mask
                temp_file_path = dataset_dest_prefix(args, prefix, 'mask')
                ds['mask'].merge_file_(temp_file_path)
                remove_mmap_idx(temp_file_path)

        ds['data'].finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
        ds['ext'].finalize(dataset_dest_file(args, output_prefix, 'ext', "idx"))
        for cls in tranv_trans.IDS_CLS:
            ds['ids'][cls].finalize(dataset_dest_file(args, output_prefix, cls, "idx"))
        ds['mask'].finalize(dataset_dest_file(args, output_prefix, 'mask', "idx"))

        LOGGER.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                src_dict.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            file_writers = {
                'raw': open(file_name(input_prefix, lang), 'r', encoding="utf-8"),
                'data': open(dest_path(args['preprocess']['destdir'], output_prefix, lang), 'w'),
                'ext': open(dest_path(args['preprocess']['destdir'], output_prefix, 'ext'), 'w'),
            }
            if args['preprocess']['id_type']:
                file_writers['ids'] = {
                    cls: open(dest_path(args['preprocess']['destdir'], output_prefix, cls), 'w')
                    for cls in tranv_trans.IDS_CLS
                }
            if args['preprocess']['rel_mask']:
                file_writers['mask'] = open(dest_path(args['preprocess']['destdir'], output_prefix, 'mask'), 'w')

            _make_dataset_func = lambda line: make_raw_dataset(
                line, ids=bool(args['preprocess']['id_type']), rel_mask=bool(args['preprocess']['rel_mask'])
            )

            def raw_write(result):
                for dfs_tokens, ext, ids, mask in itertools.chain(*result):
                    print(json.dumps(dfs_tokens), file=file_writers['data'])
                    print(ext, file=file_writers['ext'])
                    if ids is not None:
                        for cls in tranv_trans.IDS_CLS:
                            cls_ids = ids[cls]
                            if len(cls_ids) == 0:
                                cls_ids = '-1'
                            else:
                                cls_ids = ' '.join(str(id) for id in cls_ids)
                            print(cls_ids, file=file_writers['ids'][cls])
                    if mask is not None:
                        print(' '.join(itertools.chain(*mask)), file=file_writers['mask'])

            with PPool() as thread_pool:
                batch_data = []
                for line in file_writers['raw']:
                    batch_data.append(line.rstrip('\n'))
                    if len(batch_data) >= MAX_BATCH_SIZE:
                        result = thread_pool.feed(_make_dataset_func, batch_data, one_params=True)
                        raw_write(result)
                        del batch_data, result
                        batch_data = []

                if len(batch_data) > 0:
                    result = thread_pool.feed(_make_dataset_func, batch_data, one_params=True)
                    raw_write(result)
                    del batch_data, result

        else:
            make_mmap_dataset(src_dict, mask_dict, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, args['preprocess']['trainpref'], "train", lang,
                         num_workers=args['preprocess']['workers'])
        if args['preprocess']['validpref']:
            for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
        if args['preprocess']['testpref']:
            for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])

    make_all(args['preprocess']['source_lang'], src_dict)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
