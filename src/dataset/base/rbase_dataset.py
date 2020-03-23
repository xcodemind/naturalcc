# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.data import *

import glob
import ujson

from src.utils.utils import *
from src.utils.util_data import *
from src.utils.util_file import *


def load_file(filename: str) -> List:
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def load_data(dataset_files: Dict, ) -> Dict:
    data = {key: [] for key in dataset_files.keys()}
    for key, filenames in dataset_files.items():
        for fl in filenames:
            data[key].extend(load_file(fl))
    return data


class rBaseDataset(object):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)

    @classmethod
    def multiprocessing(cls, **params: Dict, ) -> Any:
        instance = cls(
            file_dir=params.get('file_dir'),
            data_lng=params.get('data_lng'),
            code_modalities=params.get('code_modalities'),
            mode=params.get('mode'),
            token_dicts=params.get('token_dicts'),
            leaf_path_k=params.get('leaf_path_k', None),  # for path
        )
        return instance

    def _construct(self, file_dir: str, data_lng: str, code_modalities: List[str], mode: str, token_dicts: TokenDicts,
                   leaf_path_k=None,  # for path
                   ) -> None:
        # load data and save data info
        self.mode = mode.lower()
        self.lng = data_lng.lower()
        self.code_modalities = code_modalities
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

        # load dataset files
        self.data = load_data(self.dataset_files)

        # some method are [], we cannot find it, so use these code as substitution
        if 'method' in self.code_modalities:
            for ind, method in enumerate(self.data['method']):
                if len(method) == 0:
                    self.data['method'][ind] = ['<no_func>']

        # check all data are same size
        size = len(self.data['comment'])
        for key, value in self.data.items():
            assert size == len(value), Exception('{} data: {}, but others: {}'.format(key, len(value), size))

        if self.mode == 'test':
            self.data['case_study'] = list(zip(self.data['code'], self.data['docstring']))
            self.data.pop('code')
            self.data.pop('docstring')
        self.LENGTH = len(self.data['index'])

        if 'code_tokens' in self.code_modalities:
            for ind, code in enumerate(self.data['code_tokens']):
                self.data['code_tokens'][ind] = token_dicts['code_tokens'].to_indices(code, UNK_WORD)
        else:
            self.data['code_tokens'] = None

        if 'method' in self.code_modalities:
            for ind, code in enumerate(self.data['method']):
                self.data['method'][ind] = token_dicts['method'].to_indices(code, UNK_WORD)
        else:
            self.data['method'] = None

        # ast modality
        if 'ast' in self.code_modalities:
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
        for ind, comment in enumerate(self.data['docstring_tokens']):
            self.data['docstring_tokens'][ind] = token_dicts['docstring_tokens'].to_indices(comment, UNK_WORD)

    @property
    def size(self):
        return self.LENGTH

    def __len__(self):
        return self.LENGTH

    def __getitem__(self, index: int) -> Dict:
        sample = {}
        if 'code_tokens' in self.data:
            sample['code_tokens'] = self.data['code_tokens'][index]
        if self.data['method'] is not None:
            sample['method'] = self.data['method'][index]
        if self.data['ast'] is not None:
            sample['ast'] = self.data['ast'][index]
        if self.data['path'] is not None:
            sample['path'] = self.data['path'][index]

        # case study
        case_study_data = self.data['case_study'][index] if 'case_study' in self.data else None
        sample['raw_index'] = self.data['index'][index]
        sample['others'] = [self.data['docstring_tokens'][index], case_study_data,
                            index, self.code_modalities, self.LENGTH, ]

        return sample


def rbase_collate_fn(batch_data: List) -> Dict:
    code_modalities, LENGTH = batch_data[0]['others'][-2], batch_data[0]['others'][-1]

    # Build and return our designed batch (dict)
    store_batch = {}

    if 'code_tokens' in code_modalities:
        # release tok modal
        tok = [batch['code_tokens'] for batch in batch_data]
        tok, tok_len, tok_mask = pad_seq(tok, include_padding_mask=True)  # pad tok
        tok, tok_len, tok_mask = to_torch_long(tok), to_torch_long(tok_len), \
                                 torch.from_numpy(tok_mask).float()
        store_batch['code_tokens'] = [tok, tok_len, tok_mask]

    if 'method' in code_modalities:
        # release method modal
        method = [batch['method'] for batch in batch_data]
        method, method_len, method_mask = pad_seq(method, include_padding_mask=True)  # pad method
        method, method_len, method_mask = to_torch_long(method), to_torch_long(method_len), \
                                          torch.from_numpy(method_mask).float()
        store_batch['method'] = [method, method_len, method_mask]

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
    comment, case_study_data, index, _, _ = zip(*[batch['others'] for batch in batch_data])
    comment, comment_len, comment_mask = pad_seq(comment, include_padding_mask=True)
    # comment to tensor
    comment, comment_len, comment_mask = to_torch_long(comment), to_torch_long(comment_len), \
                                         torch.from_numpy(comment_mask).float()
    # feed comment
    store_batch['docstring_tokens'] = [comment, comment_len, comment_mask, ]
    const_ind = np.setdiff1d(range(LENGTH), index)
    store_batch['index'] = [torch.Tensor(index).long(), torch.from_numpy(const_ind)]
    store_batch['case_study'] = case_study_data

    return store_batch


if __name__ == '__main__':
    import socket

    if socket.gethostname() in ['rtx8000']:
        run_dir = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/run/retrieval/unilang/ahn'
    elif socket.gethostname() in ['DESKTOP-ACKVLMR']:
        run_dir = r'C:\Users\GS65_2070mq\Documents\GitHub\naturalcodev2\run\retrieval\unilang\ahn'
    else:
        raise NotImplementedError
    yaml_file = os.path.join(run_dir, 'ruby.yml')
    config = load_config(yaml_file)
    CUR_SCRIPT = sys.argv[0].split('/')[-1].split('.')[0]
    LOGGER.info("Start {}.py ... =>  PID: {}".format(CUR_SCRIPT, os.getpid()))
    LOGGER.info('Load arguments in {}'.format(yaml_file))

    file_dir = config['dataset']['dataset_dir']
    data_lng = config['dataset']['source']['dataset_lng'][0]
    code_modalities = config['training']['code_modalities']
    mode = config['dataset']['source']['mode'][0]
    token_dicts = TokenDicts(config['dicts'])
    leaf_path_k = config['dataset']['leaf_path_k']

    params = {
        'file_dir': file_dir,
        'data_lng': data_lng,
        'code_modalities': code_modalities,
        'mode': mode,
        'token_dicts': token_dicts,
        'leaf_path_k': leaf_path_k,
    }

    dataset = rBaseDataset.multiprocessing(**params)

    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True,
                             collate_fn=rbase_collate_fn, )
    data_iter = iter(data_loader)
    batch_data = data_iter.__next__()
    batch_data = batch_to_cuda(batch_data)
    print(batch_data.keys())
