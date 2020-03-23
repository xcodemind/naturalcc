# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *

from joblib import Parallel, delayed
from multiprocessing import cpu_count

from multiprocessing import Pool
from src.data import *
from src.dataset.base import *
from src.utils.utils import *
from src.utils.util_data import *
from src.utils.util_file import *


class UnilangDataloader(object):
    __slots__ = ('batch_size', 'modes', 'lng', 'token_dicts', 'data_loaders', 'LENGTH',)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)


    def _construct(self,
                   # base parameters
                   base_dataset: Union[sBaseDataset, rBaseDataset],
                   collate_fn: Union[sbase_collate_fn, rbase_collate_fn],
                   params: Dict,

                   # loader parameters
                   batch_size: int,
                   modes: List[str],
                   thread_num: int,
                   ) -> None:
        self.batch_size = batch_size
        self.modes = modes

        self.lng = params.get('data_lng')
        self.LENGTH = {}
        self.data_loaders = {}
        self._load_dict(params.get('token_dicts'))
        params['token_dicts'] = self.token_dicts
        LOGGER.info("before _load_data in UnilangDataloader")
        self._load_data(thread_num, base_dataset, collate_fn, params)

    def _load_data(self,
                   thread_num: int,
                   base_dataset: Union[sBaseDataset, rBaseDataset],
                   collate_fn: Union[sbase_collate_fn, rbase_collate_fn],
                   params: Dict,
                   ) -> None:
        # 18.2358
        LOGGER.debug('read unilang dataset loader')
        LOGGER.info('read unilang dataset loader')
        paralleler = Parallel(len(self.modes))
        datasets = paralleler(delayed(base_dataset)(**dict(params, **{'mode': mode})) for mode in self.modes)
        LOGGER.info("after paralleler in  UnilangDataloader's  _load_data")
        for mode, ds in zip(self.modes, datasets):
            assert mode == ds.mode
            self.LENGTH[mode] = ds.size
            data_loader = DataLoader(
                ds, batch_size=self.batch_size,
                shuffle=False if mode == 'test' else True,
                collate_fn=collate_fn,
                num_workers=thread_num,
            )
            self.data_loaders[mode] = data_loader
        LOGGER.info("UnilangDataloader _load_data finished")
        # # slow, but for debug
        # for mode in self.modes:
        #     dataset = base_dataset(**dict(params, **{'mode': mode}))
        #     self.LENGTH[mode] = dataset.size
        #     data_loader = DataLoader(
        #         dataset, batch_size=self.batch_size,
        #         shuffle=False if mode == 'test' else True,
        #         collate_fn=collate_fn,
        #         num_workers=thread_num,
        #     )
        #     self.data_loaders[mode] = data_loader

    def _load_dict(self, token_dicts: Union[TokenDicts, Dict], ) -> None:
        if isinstance(token_dicts, dict):
            self.token_dicts = TokenDicts(token_dicts)
        elif isinstance(token_dicts, TokenDicts):
            self.token_dicts = token_dicts
        else:
            raise NotImplementedError('{}} token_dicts is wrong'.format(self.__class__.__name__))

    def __getitem__(self, key: str) -> Any:
        return self.data_loaders[key]

    @property
    def size(self, ) -> Dict:
        return self.LENGTH

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '{}: {} - {}, batch_size({})'.format(
            self.__class__.__name__, self.lng, self.LENGTH, self.batch_size
        )


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

    # parameters
    file_dir = config['dataset']['dataset_dir']
    data_lng = config['dataset']['source']['dataset_lng'][0]
    code_modalities = config['training']['code_modalities']
    modes = config['dataset']['source']['mode']
    batch_size = config['training']['batch_size']
    token_dicts = config['dicts']
    thread_num = config['common']['thread_num']

    portion = config['dataset']['portion']
    merge_xlng = config['dataset']['merge_xlng']
    leaf_path_k = config['dataset']['leaf_path_k']
    pointer_gen = config['training']['pointer']

    params = {
        'file_dir': file_dir,
        'data_lng': data_lng,
        'code_modalities': code_modalities,
        'token_dicts': token_dicts,
        'portion': portion,
        'pad_ast_sub_node': None,
        'leaf_path_k': leaf_path_k,
        'pointer_gen': pointer_gen,
    }
    dataset = UnilangDataloader(
        sBaseDataset, sbase_collate_fn, params,
        batch_size, modes, thread_num,
    )

    # batch_size = config['training']['batch_size']
    # modes = config['dataset']['source']['mode']
    # thread_num = config['common']['thread_num']
    # file_dir = config['dataset']['dataset_dir']
    # data_lng = config['dataset']['source']['dataset_lng'][0]
    # code_modalities = config['training']['code_modalities']
    # mode = config['dataset']['source']['mode'][1]
    # token_dicts = TokenDicts(config['dicts'])
    # leaf_path_k = config['dataset']['leaf_path_k']
    #
    # params = {
    #     'file_dir': file_dir,
    #     'data_lng': data_lng,
    #     'code_modalities': code_modalities,
    #     'mode': mode,
    #     'token_dicts': token_dicts,
    #     'leaf_path_k': leaf_path_k,
    # }
    # dataset = UnilangDataloader(
    #     rBaseDataset, rbase_collate_fn, params,
    #     batch_size, modes, thread_num,
    # )

    data_iter = iter(dataset['train'])
    batch_data = data_iter.__next__()
    batch_data = batch_to_cuda(batch_data)
    from pprint import pprint

    pprint(batch_data)
