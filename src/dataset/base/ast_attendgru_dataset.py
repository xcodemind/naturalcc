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


def load_data(dataset_files: Dict,mode:str ) -> Dict:
    data = {key: [] for key in dataset_files.keys()}
    for key, filenames in dataset_files.items():
        for fl in filenames:
            LOGGER.info("mode:{} keys: {}/{} file: {}/{}".format(mode , key,dataset_files.keys(),filenames.index(fl),len(filenames)))
            data[key].extend(load_file(fl))
    return data


class AstAttendGruDataset(object):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)

    # @classmethod
    # def multiprocessing(cls, **params: Dict, ) -> Any:
    #     instance = cls(
    #         file_dir=params.get('file_dir'),
    #         data_lng=params.get('data_lng'),
    #         code_modalities=params.get('code_modalities'),
    #         mode=params.get('mode'),
    #         token_dicts=params.get('token_dicts'),
    #         portion=params.get('portion', None),  # for maml
    #         pad_ast_sub_node=params.get('pad_ast_sub_node', None),  # for dtrl
    #         leaf_path_k=params.get('leaf_path_k', None),  # for path
    #         pointer_gen=params.get('pointer_gen', False),  # pointer_gen for decoder
    #     )
    #     return instance

    def _construct(self, file_dir: str, data_lng: str, code_modalities: List[str],
                   mode: str, token_dicts: TokenDicts,
                   portion=None,  # for maml
                   max_comment_len=None,
                   max_tok_len=None,
                   max_sbtao_len=None
                   ) -> None:
        # load data and save data info
        self.mode = mode.lower()
        self.lng = data_lng.lower()
        self.code_modalities = code_modalities

        self.max_comment_len = max_comment_len
        self.max_tok_len = max_tok_len
        self.max_sbtao_len = max_sbtao_len

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
        LOGGER.info("self.mode: {}".format(self.mode))
        self.data = load_data(self.dataset_files,mode=self.mode)
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


        for ind, code in enumerate(self.data['tok']):
            self.data['tok'][ind] = token_dicts['tok'].to_indices(code, UNK_WORD)
        for ind, code in enumerate(self.data['sbtao']):
            self.data['sbtao'][ind] = token_dicts['sbtao'].to_indices(code, UNK_WORD)



        # comment
        self.data['raw_comment'] = deepcopy(self.data['comment'])  # time-consuming, if not necessary, pls delete

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

        comment_extend_vocab = None

        case_study_data = self.data['case_study'][index] if 'case_study' in self.data else None

        sample = {}
        if 'tok' in self.data:
            # tok modal
            code = self.data['tok'][index]
            code_dict_comment, code_oovs = None, None
            sample['tok'] = [code, code_dict_comment, code_oovs, ]
        if 'sbtao' in self.data:
            # sbtao modal
            code = self.data['sbtao'][index]
            code_dict_comment, code_oovs = None, None
            sample['sbtao'] = [code, code_dict_comment, code_oovs, ]


        sample['others'] = [comment, comment_extend_vocab, raw_comment,
                            case_study_data, index, self.code_modalities,
                            self.max_comment_len , self.max_tok_len ,  self.max_sbtao_len  ]

        return sample


def ast_attendgru_collate_fn(batch_data: List) -> Dict:
    # 不 sort，模型里也不sort，直接对于 tok，sbtao, comment pad到数据集设置的长度
    _, _, _, _, _, code_modalities, max_comment_len , max_tok_len , max_sbtao_len = batch_data[0]['others']
    batch_size = len(batch_data)
    # if 'tok' in batch_data:
        # sort batch data, if tok available
        # batch_data.sort(key=lambda batch: len(batch['tok'][0]), reverse=True)

    # Build and return our designed batch (dict)
    store_batch = {}

    # release comment first for copy-generator
    comment, _, raw_comment, case_study_data, index, _, _,_,_ = \
        zip(*[batch['others'] for batch in batch_data])
    comment, comment_input, comment_target, comment_len = pad_comment_ast_attendgru(comment,length=max_comment_len)

    # comment to tensor
    comment, comment_input, comment_target, comment_len, \
        = map(to_torch_long, (comment, comment_input, comment_target, comment_len, ))
    # feed comment
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len, raw_comment]

    if 'tok' in batch_data[0]:
        # release tok modal
        code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code, code_len, code_mask = pad_seq_given_len(code,length=max_tok_len, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]

        store_batch['pointer'] = [None, None, None, None]

    if 'sbtao' in batch_data[0]:
        code, code_dict_comment, code_oovs = zip(*[batch['sbtao'] for batch in batch_data])
        code, code_len, code_mask = pad_seq_given_len(code,length=max_sbtao_len, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['sbtao'] = [code, code_len, code_mask]



    # other info
    store_batch['index'] = torch.Tensor(index).long()
    # store_batch['parsed_code'] = parsed_code
    store_batch['case_study'] = case_study_data
    return store_batch


def ast_attendgruv2_collate_fn(batch_data: List) -> Dict:
    # 不 sort，模型里也不sort，直接对于 tok，sbtao, comment pad到数据集设置的长度
    _, _, _, _, _, code_modalities, max_comment_len , max_tok_len , max_sbtao_len = batch_data[0]['others']
    batch_size = len(batch_data)
    # if 'tok' in batch_data:
        # sort batch data, if tok available
        # batch_data.sort(key=lambda batch: len(batch['tok'][0]), reverse=True)

    # Build and return our designed batch (dict)
    store_batch = {}

    # release comment first for copy-generator
    comment, _, raw_comment, case_study_data, index, _, _,_,_ = \
        zip(*[batch['others'] for batch in batch_data])
    comment, comment_input, comment_target, comment_len =\
        pad_comment_ast_attendgru(comment,length=max_comment_len)

    # comment to tensor
    comment, comment_input, comment_target, comment_len, \
        = map(to_torch_long, (comment, comment_input, comment_target, comment_len, ))
    # feed comment
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len, raw_comment]

    if 'tok' in batch_data[0]:
        # release tok modal
        code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code,include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]

        store_batch['pointer'] = [None, None, None, None]

    if 'sbtao' in batch_data[0]:
        code, code_dict_comment, code_oovs = zip(*[batch['sbtao'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['sbtao'] = [code, code_len, code_mask]



    # other info
    store_batch['index'] = torch.Tensor(index).long()
    # store_batch['parsed_code'] = parsed_code
    store_batch['case_study'] = case_study_data
    return store_batch

def ast_attendgruv3_collate_fn(batch_data: List) -> Dict:
    # 不 sort，模型里也不sort，直接对于 tok，sbtao, comment pad到数据集设置的长度
    _, _, _, _, _, code_modalities, max_comment_len , max_tok_len , max_sbtao_len = batch_data[0]['others']
    # batch_size = len(batch_data)
    # if 'tok' in batch_data:
        # sort batch data, if tok available
        # batch_data.sort(key=lambda batch: len(batch['tok'][0]), reverse=True)

    # Build and return our designed batch (dict)
    store_batch = {}

    # release comment first for copy-generator
    comment, _, raw_comment, case_study_data, index, _, _,_,_ = \
        zip(*[batch['others'] for batch in batch_data])
    # comment, comment_input, comment_target, comment_len =\
    #     pad_comment_ast_attendgru(comment,length=max_comment_len)

    comment, comment_input, comment_target, comment_len = pad_comment_sum(comment)

    # comment to tensor
    comment, comment_input, comment_target, comment_len, \
        = map(to_torch_long, (comment, comment_input, comment_target, comment_len, ))
    # feed comment
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len, raw_comment]

    if 'tok' in batch_data[0]:
        # release tok modal
        code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code,include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]

        store_batch['pointer'] = [None, None, None, None]

    if 'sbtao' in batch_data[0]:
        code, code_dict_comment, code_oovs = zip(*[batch['sbtao'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['sbtao'] = [code, code_len, code_mask]



    # other info
    store_batch['index'] = torch.Tensor(index).long()
    # store_batch['parsed_code'] = parsed_code
    store_batch['case_study'] = case_study_data
    return store_batch