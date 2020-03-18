# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.data import *

import glob
import random
import ujson
import math

from src.utils.util import *
from src.utils.util_data import *
from src.utils.util_file import *
from src.data import _Dict
from src.utils.util_data import build_graph


def load_file(filename: str) -> List:
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def load_data(dataset_files: Dict, ) -> Dict:
    data = {key: [] for key in dataset_files.keys()}
    for key, filenames in dataset_files.items():
        for fl in filenames:
            data[key].extend(load_file(fl))
    return data


def get_tree_max_pad_len(dst_dir: str, cur_lng: str) -> Tuple:
    while True:
        try:
            with open(os.path.join(dst_dir, 'info.txt'), 'r') as reader:
                max_pad_len_dict = ujson.loads(reader.read())
                cur_pad_len = max_pad_len_dict[cur_lng]
                max_pad_len = max(max_pad_len_dict.values())
                break
        except:
            continue
    return cur_pad_len, max_pad_len,


class gBaseDataset(object):
    # __slots__ = ('mode', 'lng', 'code_modalities', 'LENGTH', 'dataset_files', 'data',
    #              'case_study_files', 'case_study_data', )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)

    def _construct(self, file_dir: str, data_lng: str, code_modalities: List[str], mode: str, token_dicts: TokenDicts,
                   portion=None,  # for maml
                   pad_ast_sub_node=None,  # for dtrl
                   leaf_path_k=None,  # for path
                   pointer_gen=False,  # pointer_gen for decoder
                   ) -> None:
        # load data and save data info
        self.mode = mode.lower()
        self.lng = data_lng.lower()
        self.code_modalities = code_modalities
        self.pointer_gen = pointer_gen
        # LOGGER.debug(self.code_modalities)
        assert self.mode in MODES, \
            Exception('{} mode is not in {}'.format(self.mode, MODES, ))

        load_keys = self.code_modalities + ['index', 'comment', ]
        # if test, load [code/docstring] as case-study info
        if self.mode == 'test':
            load_keys.extend(['code', 'docstring'])
        LOGGER.debug('{} load {}'.format(self.mode, load_keys))
        self.dataset_files = {
            key: sorted([filename for filename in
                         glob.glob('{}/*'.format(os.path.join(file_dir, data_lng, key)))
                         if self.mode in filename])
            for key in load_keys
        }

        # for debug train, only load 1st file
        if self.mode == 'train':
            for key, filenames in self.dataset_files.items():
                self.dataset_files[key] = filenames[:1]
        LOGGER.debug(self.dataset_files)

        # load dataset files
        self.data = load_data(self.dataset_files)
        if self.mode == 'train':
            self.data['comment_fake'] = load_file(os.path.join(file_dir, self.lng, 'comment_fake',
                                                               '{}_{}_0.txt'.format(self.lng, self.mode)))
            ########################################################################
            # for debug fake_comment, error
            self.data['comment_fake'] = self.data['comment_fake'][:len(self.data['comment'])]
            ########################################################################
        else:
            self.data['comment_fake'] = [None] * len(self.data['comment'])

        # check all data are same size
        size = len(self.data['comment'])
        for key, value in self.data.items():
            assert size == len(value), Exception('{} data: {}, but others: {}'.format(key, len(value), size))

        if self.mode == 'test':
            self.data['case_study'] = list(zip(self.data['code'], self.data['docstring']))
            self.data.pop('code')
            self.data.pop('docstring')
        # sample portion
        # LOGGER.debug(portion)
        if self.mode == 'train':
            if portion is None or portion == 1.0:
                self.LENGTH = len(self.data['index'])
            elif 0.0 < portion < 1.0:
                self.LENGTH = int(len(self.data['index']) * portion)
                random_indices = random.sample(range(len(self.data['index'])), self.LENGTH)
                for key, value in self.data.items():
                    self.data[key] = [value[ind] for ind in random_indices]
            else:
                raise NotImplementedError('portion can only be (0, 1]')
        else:
            self.LENGTH = len(self.data['index'])

        # tok modality with copy mech., tok/sbt only
        # self.data['parsed_code'] = deepcopy(self.data['tok'])  # for eval
        if 'tok' in self.code_modalities:
            self.data['code'] = self.data['tok']
            code_key = 'tok'
            self.data['sbt'] = None
            self.data['sbt2'] = None
        elif 'sbt' in self.code_modalities:
            self.data['code'] = self.data['sbt']
            code_key = 'sbt'
            self.data['tok'] = None
            self.data['sbt2'] = None
        elif 'sbt2' in self.code_modalities:
            self.data['code'] = self.data['sbt2']
            code_key = 'sbt2'
            self.data['tok'] = None
            self.data['sbt'] = None
        else:
            code_key = None
            self.data['tok'] = None
            self.data['sbt'] = None
            self.data['sbt2'] = None
        if code_key is not None:
            if self.pointer_gen:
                # oov
                # parse code for oov
                self.data['code_dict_comment'] = [None] * len(self.data['code'])
                self.data['code_oovs'] = [None] * len(self.data['code'])
                for ind, code in enumerate(self.data['code']):
                    extend_vocab, oovs = extend_dict(code, token_dicts['comment'])
                    self.data['code_dict_comment'][ind] = extend_vocab
                    self.data['code_oovs'][ind] = oovs
                    self.data['code'][ind] = token_dicts[code_key].to_indices(code, UNK_WORD)
            else:
                for ind, code in enumerate(self.data['code']):
                    self.data['code'][ind] = token_dicts[code_key].to_indices(code, UNK_WORD)

        # ast modality
        if 'ast' in self.code_modalities:
            if merge_xlng:
                cur_len, max_len, = get_tree_max_pad_len(file_dir, self.lng)
                if cur_len != max_len:
                    self.data['ast'], _ = pad_tree_to_max(self.data['ast'], max_len, )
            else:
                pass

            self.data['ast'] = [build_graph(tree_dict, token_dicts['ast'], ) for tree_dict in self.data['ast']]
        else:
            self.data['ast'] = None

        # ast path mmodality
        if 'path' in self.code_modalities:
            for ind, path in enumerate(self.data['path']):
                if leaf_path_k is None:
                    head, center, tail = zip(*path)
                else:
                    head, center, tail = zip(*path[:leaf_path_k])
                self.data['path'][ind] = [
                    [token_dicts['border'].to_indices(subtoken, UNK_WORD) for subtoken in head],
                    [token_dicts['center'].to_indices(subtoken, UNK_WORD) for subtoken in center],
                    [token_dicts['border'].to_indices(subtoken, UNK_WORD) for subtoken in tail],
                ]
        else:
            self.data['path'] = None

        # comment
        self.data['raw_comment'] = deepcopy(self.data['comment'])  # time-consuming, if not necessary, pls delete
        self.data['raw_comment_fake'] = deepcopy(
            self.data['comment_fake'])  # time-consuming, if not necessary, pls delete
        # if 'tok' in self.code_modalities:
        #     # pointer
        #     self.data['comment_extend_vocab'] = [None] * len(self.data['comment'])
        #     for ind, comment in enumerate(self.data['comment']):
        #         extend_vocab = extend_dict_with_oovs(comment, token_dicts['comment'], self.data['code_oovs'][ind])
        #         self.data['comment_extend_vocab'][ind] = extend_vocab
        #         self.data['comment'][ind] = token_dicts['comment'].to_indices(comment, UNK_WORD)
        # else:
        #     for ind, comment in enumerate(self.data['comment']):
        #         self.data['comment'][ind] = token_dicts['comment'].to_indices(comment, UNK_WORD)

        for ind, (real_comment, fake_comment) in enumerate(zip(self.data['comment'], self.data['comment_fake'])):
            self.data['comment'][ind] = token_dicts['comment'].to_indices(real_comment, UNK_WORD)
            self.data['comment_fake'][ind] = None if fake_comment is None else \
                token_dicts['comment'].to_indices(fake_comment, UNK_WORD)

    @property
    def size(self):
        return self.LENGTH

    def __len__(self):
        return self.LENGTH

    def __getitem__(self, index: int) -> Dict:
        raw_comment = self.data['raw_comment'][index]
        raw_comment_fake = self.data['raw_comment_fake'][index]
        comment = self.data['comment'][index]
        fake_comment = self.data['comment_fake'][index]
        # comment_extend_vocab = None
        # if 'tok' in self.code_modalities:
        #     comment_extend_vocab = self.data['comment_extend_vocab'][index]
        #     code_dict_comment = self.data['code_dict_comment'][index]
        #     code_oovs = self.data['code_oovs'][index]
        # parsed_code = self.data['parsed_code'][index]
        # case study
        case_study_data = self.data['case_study'][index] if 'case_study' in self.data else None

        # sample = []
        sample = {}
        if 'tok' in self.code_modalities:
            # tok modal
            code = self.data['code'][index]
            # sample['tok'] = [code, code_dict_comment, code_oovs, ]
            # sample.extend([code, code_dict_comment, code_oovs, ])
            sample['tok'] = code
        if 'ast' in self.code_modalities:
            # ast modal
            ast = self.data['ast'][index]
            sample['ast'] = ast
        if 'path' in self.code_modalities:
            # ast path
            sample['path'] = self.data['path'][index]

        # sample['others'] = [comment, comment_extend_vocab, raw_comment,
        #                     case_study_data, parsed_code, index, self.code_modalities, ]
        sample['others'] = [comment, raw_comment, fake_comment, raw_comment_fake,
                            case_study_data, index, self.code_modalities, ]

        return sample


def gbase_collate_fn(batch_data: List) -> Dict:
    code_modalities = batch_data[0]['others'][-1]
    batch_size = len(batch_data)
    if 'tok' in code_modalities:
        # sort batch data, if tok available
        batch_data.sort(key=lambda batch: len(batch['tok']), reverse=True)

    # Build and return our designed batch (dict)
    store_batch = {'comment': [], 'pointer': [], 'case_study': []}
    for modal in code_modalities:
        store_batch[modal] = []

    # release comment first for copy-generator
    # comment
    # comment, comment_extend_vocab, raw_comment, case_study_data, parsed_code, index, _ = \
    #     zip(*[batch['others'] for batch in batch_data])
    comment, raw_comment, fake_comment, raw_comment_fake, case_study_data, index, _ = \
        zip(*[batch['others'] for batch in batch_data])

    comment, comment_input, comment_target, comment_len = pad_comment_sum(comment)
    comment, comment_input, comment_target, comment_len = map(to_torch_long, \
                                                              (comment, comment_input, comment_target, comment_len,))
    comment_label = torch.ones(comment.size(0)).long()
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len,
                              comment_label, raw_comment]
    if fake_comment[0] is None:
        store_batch['comment_fake'] = None
    else:
        fake_comment, fake_comment_input, fake_comment_target, fake_comment_len = pad_comment_sum(fake_comment)
        fake_comment, fake_comment_input, fake_comment_target, fake_comment_len, = \
            map(to_torch_long, \
                (fake_comment, fake_comment_input, fake_comment_target, fake_comment_len,))
        fake_comment_label = torch.zeros(comment.size(0)).long()
        store_batch['comment_fake'] = [fake_comment, fake_comment_input, fake_comment_target, fake_comment_len,
                                       fake_comment_label, raw_comment_fake]

    if 'tok' in code_modalities:
        # release tok modal
        # code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code = [batch['tok'] for batch in batch_data]
        code, code_len, code_mask = pad_seq(code, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]

        # # for pointer
        # pointer_extra_zeros = torch.zeros(batch_size, max([len(x) for x in code_oovs]))
        # code_dict_comment, _ = pad_seq(code_dict_comment)
        # code_dict_comment = to_torch_long(code_dict_comment)
        # store_batch['pointer'] = [code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs]

    if 'ast' in code_modalities:
        # release tree/ast modal
        ast = [batch['ast'] for batch in batch_data]
        packed_graph, root_indices, node_nums, = pack_graph(ast)
        tree_mask = tree_padding_mask(node_nums)
        root_indices, node_nums, tree_mask = to_torch_long(root_indices), to_torch_long(node_nums), \
                                             to_torch_float(tree_mask)
        store_batch['ast'] = [packed_graph, root_indices, node_nums, tree_mask]

    if 'path' in code_modalities:
        # release ast-path modal
        head, center, tail = map(
            lambda batch_list: list(itertools.chain(*batch_list)),
            zip(*[batch['path'] for batch in batch_data]),
        )
        head, head_len, head_mask = pad_seq(head, include_padding_mask=True)
        head, head_len, head_mask = to_torch_long(head), to_torch_long(head_len), \
                                    to_torch_long(head_mask)

        center, center_len, center_mask = pad_seq(center, include_padding_mask=True)
        center, center_len, center_mask = to_torch_long(center), to_torch_long(center_len), \
                                          to_torch_long(center_mask)

        tail, tail_len, tail_mask = pad_seq(tail, include_padding_mask=True)
        tail, tail_len, tail_mask = to_torch_long(tail), to_torch_long(tail_len), \
                                    to_torch_long(tail_mask)

        store_batch['path'] = [
            head, head_len, head_mask,
            center, center_len, center_mask,
            tail, tail_len, tail_mask,
        ]

    # other info
    store_batch['index'] = torch.Tensor(index).long()
    # store_batch['parsed_code'] = parsed_code
    store_batch['case_study'] = case_study_data

    return store_batch


if __name__ == '__main__':
    import socket

    if socket.gethostname() in ['rtx8000']:
        run_dir = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/run/summarization/unilang/mm2seq'
    elif socket.gethostname() in ['DESKTOP-ACKVLMR']:
        run_dir = r'C:\Users\GS65_2070mq\Documents\GitHub\naturalcodev2\run\summarization\unilang\mm2seq'
    else:
        raise NotImplementedError
    yaml_file = os.path.join(run_dir, 'python.yml')
    config = load_config(yaml_file)

    CUR_SCRIPT = sys.argv[0].split('/')[-1].split('.')[0]
    LOGGER.info("Start {}.py ... =>  PID: {}".format(CUR_SCRIPT, os.getpid()))
    LOGGER.info('Load arguments in {}'.format(yaml_file))

    file_dir = config['dataset']['dataset_dir']
    data_lng = config['dataset']['source']['dataset_lng'][0]
    code_modalities = config['training']['code_modalities']
    mode = config['dataset']['source']['mode'][0]
    token_dicts = TokenDicts(config['dicts'])

    portion = config['dataset']['portion']
    merge_xlng = config['dataset']['merge_xlng']
    leaf_path_k = config['dataset']['leaf_path_k']

    dataset = gBaseDataset(file_dir, data_lng, code_modalities, mode, token_dicts,
                           portion, merge_xlng, leaf_path_k, )
    # TimeAnalyzer(sBaseDataset._construct)(sBaseDataset, file_dir, data_lng, code_modalities, mode, token_dicts,
    #                                       portion, merge_xlng, leaf_path_k,
    #                                       mp=False, )
    # TimeAnalyzer.print_stats()

    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False,
                             collate_fn=gbase_collate_fn, )
    data_iter = iter(data_loader)
    batch_data = data_iter.__next__()
    batch_data = batch_to_cuda(batch_data)
    pprint(batch_data)
