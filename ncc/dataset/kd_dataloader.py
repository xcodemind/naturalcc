import os
import json
from ncc.data import TokenDicts
from ncc import LOGGER
from ncc.dataset.base import sBaseDataset, TeacherOutputDataset, ConcatDataset, KdCodeSumDataset
from typing import Any
from torch.utils.data import DataLoader


class KDDataloader(object):
    def __init__(self, *args: Any, **kwargs: Any ) -> None:
        self.data_loaders = {}
        self._construct(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        return self.data_loaders[key]

    def _construct(self,args,config, collate_fn,base_dataset=sBaseDataset): #config, args.dataset_type, dataset_type, collate_fn, mp_pool,
        batch_size = config['training']['batch_size']
        distill = config['kd']['distill']
        portion = config['dataset']['portion']
        self.token_dicts = TokenDicts(config['dicts'] )
        kd_path = config['kd']['kd_path']
        code_modalities = config['training']['code_modalities']
        file_dir = config['dataset']['dataset_dir']
        shuffle = config['kd']['shuffle']
        code_modalities_str = '8'.join(code_modalities)
        trainer_type = args.train_mode

        if 'leaf_path_k' in config['dataset']:
            leaf_path_k = config['dataset']['leaf_path_k']
        else:
            leaf_path_k = None

        sources_epoch = config['kd']['sources_epoch']

        if distill:
            assert args.train_mode == 'train_sl'
            LOGGER.info("get sources from sources_epoch.keys()")
            self.sources = [k.split('_')[0] for k in sources_epoch.keys()]
            self.targets = config['kd']['target'] # got  in load_config_dataset_kd
        else:
            LOGGER.info("get sources from config['kd']['source']")
            self.sources = config['kd']['source']
            self.targets = config['kd']['target']

        LOGGER.info("self.sources:{}  self.targets:{} ".format(self.sources, self.targets))

        all_dataset = {}

        # for mode in modes:
        for mode in ['train','valid','test']:
            src_datasets = []
            dataset_ids = []
            dataset_names = []
            ds_idx = 0

            # if args.train_mode == 'train_kd_sl_ft':
            #     LOGGER.info("use target_domain..... ")
            #     modes = config['dataset']['target_domain']['sources']['mode']
            #     sources = config['dataset']['target_domain']['sources']['dataset_lng']
            #     if config['dataset']['target_domain']['targets'] is not None :
            #         targets = config['dataset']['target_domain']['targets']['dataset_lng']
            #     else:
            #         targets = None
            # else:
            #     LOGGER.info("use source_domain..... ")
            #     modes = config['dataset']['source_domain']['sources']['mode']
            #     if not distill :
            #         assert config['dataset']['source_domain']['source']['select_lng'] is not None
            #         sources = config['dataset']['source_domain']['source']['select_lng']
            #     else:
            #         sources = config['dataset']['source_domain']['sources']['dataset_lng']
            #     if config['dataset']['source_domain']['targets'] is not None :
            #         targets = config['dataset']['source_domain']['targets']['dataset_lng']
            #     else:
            #         targets = None
            #
            # if targets is None:
            #     targets = ['en']



            LOGGER.info("args.train_mode: {}".format(args.train_mode))
            LOGGER.info("sources: {}".format(self.sources))
            LOGGER.info("targets: {}".format(self.targets))
            LOGGER.info("sources_epoch: {}".format(sources_epoch))


            topk_idxs = []
            topk_probs = []
            expert_scores = []
            is_distill_trainset = distill and mode == 'train'
            # cnt_e = 0
            for src in self.sources:
                for tgt in self.targets:
                    def add_dataset(src, tgt):
                        if is_distill_trainset and not os.path.exists(
                                # os.path.join(config['kd']['kd_path'], '{}_{}_topk_idx_{}_{}_epoch{}.idx'.format(src, tgt, trainer_type,
                                #                                                                code_modalities_str,
                                #                                                                sources_epoch[cnt_e]))):
                                os.path.join(config['kd']['kd_path'],
                                             '{}_{}_topk_idx_{}_{}_epoch{}.idx'.format(src, tgt, trainer_type,
                                                                                       code_modalities_str,
                                                                                       sources_epoch[src+'_'+tgt]))):
                            return 0
                        # src_ds = CodeSumDataset(opt, ctg, token_dicts.code_dict, token_dicts.comment_dict)
                        # src_ds = CodeSumDataset(opt, ctg, token_dicts.code_dict, token_dicts.comment_dict)

                        src_ds = base_dataset(file_dir, src, code_modalities, mode, portion=portion,
                                              token_dicts=self.token_dicts,pointer_gen= config['training']['pointer'],
                                              leaf_path_k = leaf_path_k)

                        src_datasets.append(src_ds)
                        len_dataset = len(src_ds)
                        LOGGER.info("src {} tgt {} len(src_ds) {} ".format(src, tgt, len_dataset ))
                        # LOGGER.info("| Add dataset {} . size:{} this_dataset_name: {}".format(src, len_dataset,
                        #                                                                 "{}_{}".format(src, tgt)))
                        dataset_names.append("{}_{}".format(src, tgt))
                        for i in range(len_dataset):
                            dataset_ids.append(ds_idx)
                        if is_distill_trainset:
                            # assert self.args.data_limit == ''
                            # path = os.path.join(kd_path,
                            #                     '{}_{}_topk_idx_{}_{}_epoch{}'.format(src, tgt, trainer_type, code_modalities_str,
                            #                                                      sources_epoch[cnt_e]))
                            path = os.path.join(kd_path,
                                                '{}_{}_topk_idx_{}_{}_epoch{}'.format(src, tgt, trainer_type, code_modalities_str,
                                                                                 sources_epoch[src + '_' + tgt]))


                            LOGGER.info("src {} tgt {} load {}".format(src,tgt , path))
                            idx_tmp = TeacherOutputDataset(path)
                            LOGGER.info("len(idx_tmp): {}".format(len(idx_tmp)))
                            topk_idxs.append(idx_tmp)
                            # topk_idxs.append(TeacherOutputDataset(path))
                            # path = os.path.join(kd_path, '{}_{}_topk_prob_{}_{}_epoch{}'.format(src, tgt, trainer_type,
                            #                                                                    code_modalities_str,
                            #                                                                    sources_epoch[cnt_e]))
                            path = os.path.join(kd_path, '{}_{}_topk_prob_{}_{}_epoch{}'.format(src, tgt, trainer_type,
                                                                                               code_modalities_str,
                                                                                               sources_epoch[src + '_' + tgt]))
                            LOGGER.info("src {} tgt {} load {}".format(src,tgt , path))
                            probs_tmp = TeacherOutputDataset(path)
                            LOGGER.info("len(probs_tmp): {}".format(len(probs_tmp)))
                            topk_probs.append(probs_tmp)
                            # topk_probs.append(TeacherOutputDataset(path))
                            # expert_bleu = os.path.join(kd_path,
                            #                            'expert_bleu1_{}_{}_{}_{}_epoch{}.json'.format(trainer_type,
                            #                                                                      src, tgt,
                            #                                                                      code_modalities_str,
                            #                                                                      sources_epoch[cnt_e]))

                            expert_bleu = os.path.join(kd_path,
                                                       'expert_bleu1_{}_{}_{}_{}_epoch{}.json'.format(trainer_type,
                                                                                                 src, tgt,
                                                                                                 code_modalities_str,
                                                                                                 sources_epoch[src + '_' + tgt]))
                            LOGGER.info("load {}".format(expert_bleu) )
                            expert_bleu = json.load(open(expert_bleu))
                            expert_scores.append(expert_bleu["bleu1_{}_{}".format(src, tgt)])
                        return 1

                    ds_idx += add_dataset(src, tgt)
                    # ds_idx += add_dataset(tgt, src)
                    # cnt_e += 1

            src_dataset = ConcatDataset(src_datasets)


            topk_idx_dataset = None
            topk_probs_dataset = None
            if is_distill_trainset:
                topk_idx_dataset = ConcatDataset(topk_idxs)
                topk_probs_dataset = ConcatDataset(topk_probs)
                assert len(topk_probs_dataset) == len(src_dataset), (len(topk_probs_dataset), len(src_dataset))
                assert len(topk_idx_dataset) == len(src_dataset)


            LOGGER.info("{} dataset_names: {} set(dataset_ids): {} expert_scores:{}".format(mode, dataset_names,
                                                                                      set(dataset_ids), expert_scores))
            all_dataset[mode] = KdCodeSumDataset(config,
                                                  src_dataset, expert_scores, topk_idx_dataset, topk_probs_dataset,
                                                  dataset_ids, dataset_names, is_train=mode == 'train')


        train_dataloader = DataLoader(dataset=all_dataset['train'], batch_size=batch_size,
                                                       shuffle=shuffle ,
                                                       num_workers=config['common']['thread_num'], collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset=all_dataset['valid'], batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=config['common']['thread_num'], collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset=all_dataset['test'], batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=config['common']['thread_num'], collate_fn=collate_fn)
        # train_dataset = all_dataset['train']
        # val_dataset = all_dataset['val']
        # test_dataset = all_dataset['test']

   
        # self.train_dataset = all_dataset['train']
        # self.val_dataset = all_dataset['val']
        # self.test_dataset = all_dataset['test']
        # self.train_dataloader = train_dataloader
        # self.val_dataloader  = val_dataloader
        # self.test_dataloader = test_dataloader

        self.data_loaders['train_dataset'] = all_dataset['train']
        self.data_loaders['valid_dataset'] = all_dataset['valid']
        self.data_loaders['test_dataset'] = all_dataset['test']

        self.data_loaders['train'] = train_dataloader
        self.data_loaders['valid'] = val_dataloader
        self.data_loaders['test'] = test_dataloader

 
        # return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader