# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.data import *

import glob
import random
import ujson
import math

from ncc.utils.utils import *
from ncc.utils.util_data import *
from ncc.utils.util_file import *
from ncc.data import _Dict
from ncc.utils.util_data import build_graph
from torch.utils.data import Dataset


def load_file(filename: str) -> List:
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def load_data(dataset_files: Dict, mode: str) -> Dict:
    data = {key: [] for key in dataset_files.keys()}
    for key, filenames in dataset_files.items():
        for fl in filenames:
            LOGGER.info("mode:{} keys: {}/{} file: {}/{}  ".format(mode, key, dataset_files.keys(), filenames.index(fl),
                                                                   len(filenames)))
            data[key].extend(load_file(fl))
    return data


class sBaseDataset(Dataset):

    def __init__(self, *argss: Any, **kwargs: Any) -> None:
        self._construct(*argss, **kwargs)

    @classmethod
    def multiprocessing(cls, **params: Dict, ) -> Any:
        instance = cls(
            file_dir=params.get('file_dir'),
            data_lng=params.get('data_lng'),
            code_modalities=params.get('code_modalities'),
            mode=params.get('mode'),
            token_dicts=params.get('token_dicts'),
            portion=params.get('portion', None),  # for maml
            pad_ast_sub_node=params.get('pad_ast_sub_node', None),  # for dtrl
            leaf_path_k=params.get('leaf_path_k', None),  # for path
            pointer_gen=params.get('pointer_gen', False),  # pointer_gen for decoder
        )
        return instance

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

        LOGGER.debug('read sbase dataset loader')
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

        self.data = load_data(self.dataset_files, mode=self.mode)
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
        if self.mode == 'train' and self.lng.lower() == 'ruby':
            if portion is None or portion == 1.0:
                self.LENGTH = len(self.data['index'])
            elif 0.0 < portion < 1.0:
                self.LENGTH = int(len(self.data['index']) * portion)
                random_indices = random.sample(range(len(self.data['index'])), self.LENGTH)
                for key, value in self.data.items():
                    self.data[key] = [value[ind] for ind in random_indices]
            else:
                raise NotImplementedError('portion can only be (0, 1]')
            # load dataset files
            LOGGER.info("load dataset: {}-{}-p{}".format(self.lng, self.mode, portion))
        else:
            self.LENGTH = len(self.data['index'])
            # load dataset files
            LOGGER.info("load dataset: {}-{}".format(self.lng, self.mode))

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
        elif 'sbtao' in self.code_modalities:
            self.data['code'] = self.data['sbtao']
            code_key = 'sbtao'
            self.data['tok'] = None
            self.data['sbt'] = None
            self.data['sbt2'] = None
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
            if pad_ast_sub_node is not None:
                # if cur language's ast sub-token len is max among all dataset, it's None
                self.data['ast'] = pad_tree_to_max(self.data['ast'], pad_ast_sub_node, )
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
        if self.pointer_gen and 'code' in self.data:
            # pointer
            self.data['comment_extend_vocab'] = [None] * len(self.data['comment'])
            for ind, comment in enumerate(self.data['comment']):
                extend_vocab = extend_dict_with_oovs(comment, token_dicts['comment'], self.data['code_oovs'][ind])
                self.data['comment_extend_vocab'][ind] = extend_vocab
                self.data['comment'][ind] = token_dicts['comment'].to_indices(comment, UNK_WORD)
        else:
            for ind, comment in enumerate(self.data['comment']):
                self.data['comment'][ind] = token_dicts['comment'].to_indices(comment, UNK_WORD)

    @property
    def size(self):
        return self.LENGTH

    def __len__(self):
        return self.LENGTH

    def __getitem__(self, index: int) -> Dict:
        raw_comment = self.data['raw_comment'][index]
        comment = self.data['comment'][index]
        if self.pointer_gen and 'code' in self.data:
            comment_extend_vocab = self.data['comment_extend_vocab'][index]
        else:
            comment_extend_vocab = None
        # parsed_code = self.data['parsed_code'][index]
        # case study
        case_study_data = self.data['case_study'][index] if 'case_study' in self.data else None

        sample = {}
        if 'code' in self.data:
            # tok modal
            code = self.data['code'][index]
            if self.pointer_gen:
                code_dict_comment = self.data['code_dict_comment'][index]
                code_oovs = self.data['code_oovs'][index]
            else:
                code_dict_comment, code_oovs = None, None
            sample['tok'] = [code, code_dict_comment, code_oovs, ]
        if 'ast' in self.code_modalities:
            # ast modal
            ast = self.data['ast'][index]
            sample['ast'] = ast
        if 'path' in self.code_modalities:
            # ast path
            sample['path'] = self.data['path'][index]

        # sample['others'] = [comment, comment_extend_vocab, raw_comment,
        #                     case_study_data, parsed_code, index, self.code_modalities, ]
        sample['others'] = [comment, comment_extend_vocab, raw_comment,
                            case_study_data, index, self.code_modalities, self.pointer_gen, ]

        return sample


def sbase_collate_fn(batch_data: List) -> Dict:
    code_modalities, pointer_gen = batch_data[0]['others'][-2], batch_data[0]['others'][-1]
    batch_size = len(batch_data)

    # if 'tok' in batch_data:
    #     # sort batch data, if tok available
    #     batch_data.sort(key=lambda batch: len(batch['tok'][0]), reverse=True)

    # Build and return our designed batch (dict)
    store_batch = {}

    # release comment first for copy-generator
    comment, comment_extend_vocab, raw_comment, case_study_data, index, _, _ = \
        zip(*[batch['others'] for batch in batch_data])
    comment, comment_input, comment_target, comment_len = pad_comment_sum(comment)
    if comment_extend_vocab[0] is not None:  # tuple of None
        _, _, comment_extend_vocab, _ = pad_comment_sum(comment_extend_vocab)
    else:
        comment_extend_vocab = None
    # comment to tensor
    comment, comment_input, comment_target, comment_len, comment_extend_vocab, \
        = map(to_torch_long, (comment, comment_input, comment_target, comment_len, comment_extend_vocab,))
    # feed comment
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len, raw_comment]

    if 'tok' in batch_data[0]:
        # release tok modal
        code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]
        if pointer_gen:
            # for pointer
            pointer_extra_zeros = torch.zeros(batch_size, max([len(x) for x in code_oovs]))
            code_dict_comment, _ = pad_seq(code_dict_comment)
            code_dict_comment = to_torch_long(code_dict_comment)
            store_batch['pointer'] = [code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs]
        else:
            store_batch['pointer'] = [None, None, None, None]

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
    args = load_args(yaml_file)

    CUR_SCRIPT = sys.argv[0].split('/')[-1].split('.')[0]
    LOGGER.info("Start {}.py ... =>  PID: {}".format(CUR_SCRIPT, os.getpid()))
    LOGGER.info('Load arguments in {}'.format(yaml_file))

    file_dir = args['dataset']['dataset_dir']
    # data_lng = args['dataset']['source']['dataset_lng'][0]
    data_lng = 'php'
    code_modalities = args['training']['code_modalities']
    # mode = args['dataset']['source']['mode'][-1]
    mode = 'valid'
    token_dicts = TokenDicts(args['dicts'])
    # token_dicts = None

    portion = args['dataset']['portion']
    leaf_path_k = args['dataset']['leaf_path_k']
    pointer_gen = args['training']['pointer']

    dataset = sBaseDataset.multiprocessing(**{
        'file_dir': file_dir,
        'data_lng': data_lng,
        'code_modalities': code_modalities,
        'mode': mode,
        'token_dicts': token_dicts,
        'portion': portion,
        'pad_ast_sub_node': None,
        'leaf_path_k': leaf_path_k,
        'pointer_gen': pointer_gen,
    })

    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True,
                             collate_fn=sbase_collate_fn, )
    data_iter = iter(data_loader)
    batch_data = data_iter.__next__()
    batch_to_cuda(batch_data)
