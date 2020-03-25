# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from ncc import *

from ncc.utils.utils import mpool
from ncc.utils.util_file import load_config, load_config_kd
from ncc.dataset.base import *
from ncc.dataset import *
from ncc.utils.util_name import time_id, md5_id
from ncc.model.template import *


def CONST_TIME_CODE(debug):
    return time_id(debug)


CONST_MD5_CODE = md5_id()


def build_model(config: Dict, model: IModel, ) -> Module:
    def _init_param(model):
        for p in model.parameters():
            p.data.uniform_(-config['common']['init_bound'], config['common']['init_bound'])

    def _load_pretrained_weights():
        if config['common']['init_weights'] is None or not os.path.exists(config['common']['init_weights']):
            _init_param(model)
            LOGGER.info('Initialize model weights from scratch.')
        else:
            try:
                # print("init_weights: ", config['common']['init_weights'])
                assert os.path.exists(config['common']['init_weights'])
            except Exception as err:
                LOGGER.error(str(err))
                assert False
            LOGGER.info('Loading from {}'.format(config['common']['init_weights']))
            checkpoint = torch.load(config['common']['init_weights'], map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)

    _load_pretrained_weights()

    if config['common']['device'] is not None:
        LOGGER.info('Model to GPU')
        model.cuda()
    else:
        LOGGER.info('Model to CPU')
    LOGGER.info('model structure: {}'.format(str(model)))

    return model


def get_model_path(args, config):
    # root_dir = config['dataset']['save_dir']
    if args.train_mode == 'test':
        if config['testing']['init_weights'] is None:
            model_path = None
        else:
            # model_path = os.path.join(root_dir, config['testing']['init_weights'])
            model_path = config['testing']['init_weights']
    elif args.train_mode == 'train_sc_ft':
        if config['sc']['init_weights'] is None:
            model_path = None
        else:
            # model_path = os.path.join(root_dir, config['sc']['init_weights'])
            model_path = config['sc']['init_weights']
    elif args.train_mode == 'train_sl_ft':
        if config['sl']['init_weights'] is None:
            model_path = None
        else:
            # model_path = os.path.join(root_dir, config['sl']['init_weights'])
            model_path = config['sl']['init_weights']
    else:
        model_path = None
    config['model_path'] = model_path
    return config


def build_model_kd(config: Dict, dataset_type: str, model: IModel) -> Module:
    LOGGER.info('build model for {}'.format(dataset_type))
    # model_path = config['common']['init_weights']
    model_path = config['model_path']

    def _init_param(model):
        for p in model.parameters():
            p.data.uniform_(-config['common']['init_bound'], config['common']['init_bound'])

    def _load_pretrained_weights():
        if model_path is None:
            _init_param(model)
            LOGGER.debug('Initialize model weights from scratch.')
        else:
            try:
                # print("init_weights: ", model_path)
                assert os.path.exists(model_path)
            except Exception as err:
                LOGGER.error(str(err))
                LOGGER.info("need model_path: {}".format(model_path))
                assert False
            LOGGER.info('Loading from {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)

    # try:
    #     assert config['dataset'][dataset_type] is not None
    # except Exception as err:
    #     LOGGER.error(str(err))
    #     assert False
    _load_pretrained_weights()

    if config['common']['device'] is not None:
        LOGGER.info('Model to GPU')
        model.cuda()
    else:
        LOGGER.info('Model to CPU')
    LOGGER.debug('model structure: {}'.format(str(model)))

    return model


def build_critic(config: Dict, critic: IModel, ):
    # LOGGER.info('build critic for {}'.format(dataset_type))

    def _init_param(critic):
        for p in critic.parameters():
            p.data.uniform_(-config['common']['init_bound'], config['common']['init_bound'])

    def _load_pretrained_weights():
        if config['ac']['init_weights_critic'] is None:
            _init_param(critic)
            LOGGER.debug('Initialize critic weights from scratch.')
        else:
            try:
                print("config['ac']['init_weights_critic']: ", config['ac']['init_weights_critic'])
                assert os.path.exists(config['ac']['init_weights_critic'])
            except Exception as err:
                LOGGER.error(str(err))
                assert False
            LOGGER.info('Loading from {}'.format(config['ac']['init_weights_critic']))
            checkpoint = torch.load(config['ac']['init_weights_critic'], map_location=lambda storage, loc: storage)
            critic.load_state_dict(checkpoint)

    _load_pretrained_weights()

    if config['common']['device'] is not None:
        LOGGER.info('Model to GPU')
        critic.cuda()
    else:
        LOGGER.info('Model to CPU')
    LOGGER.debug('critic structure: {}'.format(str(critic)))

    return critic


def build_disc(config: Dict, dataset_type: str, disc: IModel, ):
    LOGGER.info('build disc for {}'.format(dataset_type))

    def _init_param(disc):
        for p in disc.parameters():
            p.data.uniform_(-config['common']['init_bound'], config['common']['init_bound'])

    def _load_pretrained_weights():
        if config['gan']['init_weights_disc'] is None:
            _init_param(disc)
            LOGGER.debug('Initialize disc weights from scratch.')
        else:
            try:
                print("config['gan']['init_weights_disc']: ", config['gan']['init_weights_disc'])
                assert os.path.exists(config['gan']['init_weights_disc'])
            except Exception as err:
                LOGGER.error(str(err))
                assert False
            LOGGER.info('Loading from {}'.format(config['gan']['init_weights_disc']))
            checkpoint = torch.load(config['gan']['init_weights_disc'], map_location=lambda storage, loc: storage)
            disc.load_state_dict(checkpoint)

    try:
        assert config['dataset'][dataset_type] is not None
    except Exception as err:
        LOGGER.error(str(err))
        assert False
    _load_pretrained_weights()

    if config['common']['device'] is not None:
        LOGGER.info('Model to GPU')
        disc.cuda()
    else:
        LOGGER.info('Model to CPU')
    LOGGER.debug('disc structure: {}'.format(str(disc)))

    return disc


def build_reward_model(config: Dict, dataset_type: str, reward_model: IModel, ):
    LOGGER.info('build reward_model for {}'.format(dataset_type))

    def _init_param(reward_model):
        for p in reward_model.parameters():
            p.data.uniform_(-config['common']['init_bound'], config['common']['init_bound'])

    def _load_pretrained_weights():
        if config['arel']['init_weights_reward_model'] is None:
            _init_param(reward_model)
            LOGGER.debug('Initialize reward_model weights from scratch.')
        else:
            try:
                print("config['arel']['init_weights_reward_model']: ", config['arel']['init_weights_reward_model'])
                assert os.path.exists(config['arel']['init_weights_reward_model'])
            except Exception as err:
                LOGGER.error(str(err))
                assert False
            LOGGER.info('Loading from {}'.format(config['arel']['init_weights_reward_model']))
            checkpoint = torch.load(config['arel']['init_weights_reward_model'],
                                    map_location=lambda storage, loc: storage)
            reward_model.load_state_dict(checkpoint)

    try:
        assert config['dataset'][dataset_type] is not None
    except Exception as err:
        LOGGER.error(str(err))
        assert False
    _load_pretrained_weights()

    if config['common']['device'] is not None:
        LOGGER.info('Model to GPU')
        reward_model.cuda()
    else:
        LOGGER.info('Model to CPU')
    LOGGER.debug('reward_model structure: {}'.format(str(reward_model)))

    return reward_model


def get_args() -> namedtuple:
    import argparse

    parser = argparse.ArgumentParser(description='for run.py')
    # default rules
    # parser.add_argument('--', default=None, type=None, help='XXX', required=False)

    parser.add_argument('--yaml', default=None, type=str, help='yaml file name')
    parser.add_argument('--task', default=None, type=str, help='summarization/retrieval')
    parser.add_argument('--lang_mode', default=None, type=str, help='unilang/xlang')
    parser.add_argument('--method_name', default=None, type=str, help='mm2seq/deepcom/code2seq')
    '''
    'train_sl': train via supervise learning 
    'train_ft': train via fine-tune
    'train_pg': train via policy gradient
    'train_ac': train via action-critic
    'train_sc': train via self-critical
    'train_gan' train via generative adversarial network (GAN)

    'test': eval
    '''

    parser.add_argument('--train_mode', default=None, type=str,
                        help='train_sl/train_ft/train_pg/train_sc/train_ac/train_gan/test')
    parser.add_argument('--dataset_type', default=None, type=str, help='all/source/target')
    parser.add_argument('--multi_processing', type=int, default=0,
                        help='using multi-processing to accelerate data-loading(default True)')
    parser.add_argument('--debug', default=0, type=int, help='debug mode')
    # parser.add_argument('--occupy_gpu', default='no', type=str, help='occupy_gpu: g(0-8)-(1-48)gb')
    parser.add_argument('--occupy_gpu', default='no', type=str, help='occupy_gpu: "0-1.2" means occupy 1.2gb on gpu0')
    parser.add_argument('--save_dir', type=str, required=False)
    parser.add_argument('--dataset_dir', type=str, required=False)
    parser.add_argument('--device', type=int, default=0, required=False)
    parser.add_argument('--log_root_dir', type=str, default=None, required=False)

    ARGS = parser.parse_args()
    return ARGS


def get_args_new() -> namedtuple:
    import argparse

    parser = argparse.ArgumentParser(description='for run.py')
    # default rules
    # parser.add_argument('--', default=None, type=None, help='XXX', required=False)

    parser.add_argument('yml', type=str, help='yaml file name')

    # parser.add_argument('--task', default=None, type=str, help='summarization/retrieval')
    # parser.add_argument('--lang_mode', default=None, type=str, help='unilang/xlang')
    # parser.add_argument('--method_name', default=None, type=str, help='mm2seq/deepcom/code2seq')
    # '''
    # 'train_sl': train via supervise learning
    # 'train_ft': train via fine-tune
    # 'train_pg': train via policy gradient
    # 'train_ac': train via action-critic
    # 'train_sc': train via self-critical
    # 'train_gan' train via generative adversarial network (GAN)
    #
    # 'test': eval
    # '''
    #
    # parser.add_argument('--train_mode', default=None, type=str,
    #                     help='train_sl/train_ft/train_pg/train_sc/train_ac/train_gan/test')
    # parser.add_argument('--dataset_type', default=None, type=str, help='all/source/target')
    # parser.add_argument('--multi_processing', type=int,
    #                     help='using multi-processing to accelerate data-loading(default True)')
    # parser.add_argument('--debug', default=0, type=int, help='debug mode')
    # parser.add_argument('--occupy_gpu', default='no', type=str, help='occupy_gpu: g(0-8)-(1-48)gb')

    ARGS = parser.parse_args()
    return ARGS


def get_log_filename(args: namedtuple) -> str:
    if args.debug:
        log_filename = '{}_{}_{}_{}_{}.log'.format(
            str.upper(args.task), str.upper(args.lang_mode), str.upper(args.method_name), str.upper(args.train_mode),
            CONST_TIME_CODE(args.debug),
        )
    else:
        log_filename = '{}_{}_{}_{}_{}.{}.log'.format(
            str.upper(args.task), str.upper(args.lang_mode), str.upper(args.method_name), str.upper(args.train_mode),
            CONST_TIME_CODE(args.debug), CONST_MD5_CODE,
        )

    return log_filename


def run_init(yml_filename: str, config=None) -> Dict:
    LOGGER.info("Start {} ... =>  PID: {}".format(sys.argv[0], os.getpid()))

    yaml_file = os.path.join(sys.path[0], yml_filename)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    if config is None:
        config = load_config(yaml_file)
    else:
        config = load_config(yaml_file, config)

    LOGGER.info('# ------------ env init ------------ #')
    LOGGER.info('Init(seed {}): torch/random/numpy...'.format(config['common']['seed']))
    torch.manual_seed(config['common']['seed'])
    random.seed(config['common']['seed'])
    np.random.seed(config['common']['seed'])
    LOGGER.info("device: {}".format(config['common']['device']))
    if config['common']['device'] is not None:
        LOGGER.debug(config['common']['device'])
        torch.cuda.set_device(config['common']['device'])
    LOGGER.info('Device: {}'.format(torch.cuda.get_device_name()))
    LOGGER.info('# ------------ env init ------------ #')
    return config


def run_init_kd(yml_filename: str, config=None) -> Dict:
    LOGGER.info("Start {} ... =>  PID: {}".format(sys.argv[0], os.getpid()))

    yaml_file = os.path.join(sys.path[0], yml_filename)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    if config is None:
        config = load_config_kd(yaml_file)
    else:
        config = load_config_kd(yaml_file, config)

    LOGGER.info('# ------------ env init ------------ #')
    LOGGER.info('Init(seed {}): torch/random/numpy...'.format(config['common']['seed']))
    torch.manual_seed(config['common']['seed'])
    random.seed(config['common']['seed'])
    np.random.seed(config['common']['seed'])
    LOGGER.info("device: {}".format(config['common']['device']))
    if config['common']['device'] is not None:
        LOGGER.debug(config['common']['device'])
        torch.cuda.set_device(config['common']['device'])
    LOGGER.info('Device: {}'.format(torch.cuda.get_device_name()))
    LOGGER.info('# ------------ env init ------------ #')
    return config


def load_config_dataset(args: namedtuple,
                        dataloader: Union[XlangDataloader, UnilangDataloader],
                        base_dataset: Union[sBaseDataset, rBaseDataset],
                        collate_fn: Union[sbase_collate_fn, rbase_collate_fn], config=None) -> Tuple:
    if config is None:
        config = run_init(args.yaml)
    else:
        config = run_init(args.yaml, config=config)
    if dataloader == XlangDataloader:
        LOGGER.debug('data_loader: XlangDataloader')

        # for dtrl
        if (args.method_name == 'dtrl') and ('dtrl' in config) and \
                ('merge_xlng' in config['dtrl']) and config['dtrl']['merge_xlng']:
            LOGGER.info('DTRL train, shrink batch_size into its half')
            config['training']['batch_size'] = config['training']['batch_size'] // 2

        # because load data is time-consuming, we only load data we need
        src_dataset_info = config['dataset']['source'] \
            if (args.dataset_type == 'source') or (args.dataset_type == 'all') else None
        trg_dataset_info = config['dataset']['target'] \
            if (args.dataset_type == 'target') or (args.dataset_type == 'all') else None

        batch_size = config['training']['batch_size']
        thread_num = config['common']['thread_num']

        if 'leaf_path_k' in config['dataset']:
            leaf_path_k = config['dataset']['leaf_path_k']
        else:
            leaf_path_k = None

        if args.task == 'summarization':

            if (args.method_name == 'dtrl') and ('dtrl' in config) and \
                    ('merge_xlng' in config['dtrl']) and config['dtrl']['merge_xlng']:
                src_xlang = config['dataset']['source']['dataset_lng']
            else:
                src_xlang = None

            params = {
                'file_dir': config['dataset']['dataset_dir'],
                'code_modalities': config['training']['code_modalities'],
                'token_dicts': config['dicts'],
                'portion': config['dataset']['portion'],
                'pad_ast_sub_node': None,
                'leaf_path_k': leaf_path_k,
                'pointer_gen': config['training']['pointer'],
                'src_xlang': src_xlang,
            }

            dataset = dataloader(
                base_dataset, collate_fn, params,
                src_dataset_info, trg_dataset_info, batch_size, thread_num,
            )
        elif args.task == 'retrieval':

            params = {
                'file_dir': config['dataset']['dataset_dir'],
                'code_modalities': config['training']['code_modalities'],
                'token_dicts': config['dicts'],
                'leaf_path_k': leaf_path_k,
            }

            dataset = dataloader(
                base_dataset, collate_fn, params,
                src_dataset_info, trg_dataset_info, batch_size, thread_num,
            )


    else:
        raise NotImplementedError
    LOGGER.info(dataset)

    config['training']['token_num'] = {
        # key.split('_')[0]: token_size
        key: token_size
        for key, token_size in dataset.token_dicts.size.items()
    }
    LOGGER.debug(config)

    return config, dataset,


def load_config_dataset_ast_attendgru(args: namedtuple,
                                      dataloader: Union[XlangDataloader, UnilangDataloader],
                                      base_dataset: Union[sBaseDataset, rBaseDataset],
                                      collate_fn: Union[sbase_collate_fn, rbase_collate_fn], config=None) -> Tuple:
    if config is None:
        config = run_init(args.yaml)
    else:
        config = run_init(args.yaml, config=config)
    if dataloader == XlangDataloader:
        LOGGER.debug('data_loader: XlangDataloader')

        # for dtrl
        if (args.method_name == 'dtrl') and ('dtrl' in config) and \
                ('merge_xlng' in config['dtrl']) and config['dtrl']['merge_xlng']:
            LOGGER.info('DTRL train, shrink batch_size into its half')
            config['training']['batch_size'] = config['training']['batch_size'] // 2

        # because load data is time-consuming, we only load data we need
        src_dataset_info = config['dataset']['source'] \
            if (args.dataset_type == 'source') or (args.dataset_type == 'all') else None
        trg_dataset_info = config['dataset']['target'] \
            if (args.dataset_type == 'target') or (args.dataset_type == 'all') else None

        batch_size = config['training']['batch_size']
        thread_num = config['common']['thread_num']

        # if 'leaf_path_k' in config['dataset']:
        #     leaf_path_k = config['dataset']['leaf_path_k']
        # else:
        #     leaf_path_k = None

        if args.task == 'summarization':
            # if (args.method_name == 'dtrl') and ('dtrl' in config) and \
            #         ('merge_xlng' in config['dtrl']) and config['dtrl']['merge_xlng']:
            #     src_xlang = config['dataset']['source']['dataset_lng']
            # else:
            #     src_xlang = None

            params = {
                'file_dir': config['dataset']['dataset_dir'],
                'code_modalities': config['training']['code_modalities'],
                'token_dicts': config['dicts'],
                'portion': config['dataset']['portion'],
                'max_comment_len': config['dataset']['max_comment_len'],
                'max_tok_len': config['dataset']['max_tok_len'],
                'max_sbtao_len': config['dataset']['max_sbtao_len']

            }

            dataset = dataloader(
                base_dataset, collate_fn, params,
                src_dataset_info, trg_dataset_info, batch_size, thread_num,
            )



    else:
        raise NotImplementedError
    LOGGER.info(dataset)

    config['training']['token_num'] = {key.split('_')[0]: token_size for key, token_size in
                                       dataset.token_dicts.size.items()}
    LOGGER.info(config)
    return config, dataset,


def load_config_dataset_kd(args: namedtuple,
                           base_dataset=sBaseDataset, config=None):
    if config is None:
        config = run_init_kd(args.yaml)
    else:
        config = run_init_kd(args.yaml, config=config)

    if args.train_mode in ['train_sl_ft', 'train_sc_ft', 'test']:
        # in kd, use source and target in target domain  when finetune the multilingual student model
        config['dataset']['source'] = config['dataset']['target_domain']['source']
        config['dataset']['target'] = config['dataset']['target_domain']['target']

        LOGGER.debug('data_loader: XlangDataloader')

        # for dtrl
        if (args.method_name == 'dtrl') and ('dtrl' in config) and \
                ('merge_xlng' in config['dtrl']) and config['dtrl']['merge_xlng']:
            LOGGER.info('DTRL train, shrink batch_size into its half')
            config['training']['batch_size'] = config['training']['batch_size'] // 2

        # because load data is time-consuming, we only load data we need
        src_dataset_info = config['dataset']['source'] \
            if (args.dataset_type == 'source') or (args.dataset_type == 'all') else None
        trg_dataset_info = config['dataset']['target'] \
            if (args.dataset_type == 'target') or (args.dataset_type == 'all') else None

        batch_size = config['training']['batch_size']
        thread_num = config['common']['thread_num']

        if 'leaf_path_k' in config['dataset']:
            leaf_path_k = config['dataset']['leaf_path_k']
        else:
            leaf_path_k = None

        if args.task == 'summarization':

            if (args.method_name == 'dtrl') and ('dtrl' in config) and \
                    ('merge_xlng' in config['dtrl']) and config['dtrl']['merge_xlng']:
                src_xlang = config['dataset']['source']['dataset_lng']
            else:
                src_xlang = None

            params = {
                'file_dir': config['dataset']['dataset_dir'],
                'code_modalities': config['training']['code_modalities'],
                'token_dicts': config['dicts'],
                'portion': config['dataset']['portion'],
                'pad_ast_sub_node': None,
                'leaf_path_k': leaf_path_k,
                'pointer_gen': config['training']['pointer'],
                'src_xlang': src_xlang,
            }

            dataset = XlangDataloader(
                base_dataset, sbase_collate_fn, params,
                src_dataset_info, trg_dataset_info, batch_size, thread_num,
            )


    elif args.train_mode == 'train_sl':
        # if args.train_mode in [ 'train_kd_sl_ft','train_kd_sc_ft']:
        #     LOGGER.info("use target_domain..... ")
        #     # modes = config['dataset']['target_domain']['sources']['mode']
        #     sources = config['dataset']['target_domain']['source']['dataset_lng']
        #     if config['dataset']['target_domain']['target'] is not None:
        #         targets = config['dataset']['target_domain']['target']['dataset_lng']
        #     else:
        #         targets = None
        # else:
        LOGGER.info("use source_domain..... ")
        # modes = config['dataset']['source_domain']['sources']['mode']
        if not config['kd']['distill']:
            assert config['dataset']['source_domain']['source']['select_lng'] is not None
            sources = config['dataset']['source_domain']['source']['select_lng']
        else:
            sources = config['dataset']['source_domain']['source']['dataset_lng']
        if config['dataset']['source_domain']['target'] is not None:
            targets = config['dataset']['source_domain']['target']['dataset_lng']
        else:
            targets = None

        if targets is None:
            targets = ['en']

        config['kd']['source'] = sources
        config['kd']['target'] = targets
        dataset = KDDataloader(args, config, collate_fn=kd_codesum_collate_fn, base_dataset=base_dataset)
    # if args.multi_processing:
    #     with mpool() as mp_pool:
    #         dataset = dataloader(config, args.dataset_type, dataset_type, collate_fn, mp_pool, )
    # else:
    #     dataset = dataloader(config, args.dataset_type, dataset_type, collate_fn, None)
    # LOGGER.debug(dataset)

    # config = dataset.add_token_num(config)
    LOGGER.info(config)
    config['training']['token_num'] = {key.split('_')[0]: token_size for key, token_size in
                                       dataset.token_dicts.size.items()}
    return config, dataset


def get_save_dir(config: Dict, args: namedtuple) -> str:
    save_dir = os.path.join(
        config['dataset']['save_dir'],
        '{}_{}_{}.{}'.format(args.task, args.task_mode, CONST_TIME_CODE, CONST_MD5_CODE, ),
    )
    try:
        # os.makedirs(save_dir)
        pass
    except Exception as err:
        print(err)
        '''
        With time-stamp and md5 code, this error almost cannot be raised.
        If you indeed encounter, you are the one. You must win a lottery.
        '''
        LOGGER.error('Dir {} has already existed. Pls, wait and try again to avoid time/md5 conflict.')
    LOGGER.info('save dir: {}'.format(save_dir))
    return save_dir


class DictAsObj(dict):
    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def _row_sparse_pull(self, parameter, out, row_id, full_idx=False):
        """Internal method to invoke pull operations on KVStore. If `full_idx` is set to True,
        `kv.pull` is preferred instead of `kv.row_sparse_pull`.
        """
        # initialize kv and params if not already
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        idx = self._param2idx[parameter.name]
        if full_idx and 'dist' not in self._kvstore.type:
            assert row_id.size == out.shape[0]
            self._kvstore.pull(idx, out=out, priority=-idx, ignore_sparse=False)
        else:
            self._kvstore.row_sparse_pull(idx, out=out, row_ids=row_id, priority=-idx)
