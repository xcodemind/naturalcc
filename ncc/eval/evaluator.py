# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from ncc import *
from ncc.data import *
from ncc.metric import *
from ncc.model.template import *
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.util_eval import eval_metrics, calculate_scores_multi_dataset, eval_per_metrics, \
    normalize, ACC, MAP, MRR, NDCG, IDCG
from ncc.metric.base import *
from ncc.utils.utils import save_json
from tabulate import tabulate
from ncc.utils.constants import METRICS
from tqdm import tqdm


def load_data(model, datatype):
    import glob, ujson
    ast_files = sorted([filename for filename in glob.glob('{}/*'.format(os.path.join(
        model.config['dataset']['dataset_dir'], 'ruby', datatype
    ))) if 'test' in filename])

    len_list = []
    for fl in ast_files:
        with open(fl, 'r') as reader:
            line = reader.readline().strip()
            while len(line) > 0:
                line = ujson.loads(line)
                len_list.append(len(line))
                line = reader.readline().strip()
    return len_list

class Evaluator(object):
    def __init__(self, ) -> None:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def retrieval_eval(model: IModel, data_loader: DataLoader, pool_size=-1, ) -> Any:
        with torch.no_grad():
            model.eval()
            data_iter = iter(data_loader)  # init

            accs, mrrs, maps, ndcgs = [], [], [], []

            code_reprs, cmnt_reprs = [], []
            for iteration in range(1, 1 + len(data_loader)):
                batch = data_iter.__next__()
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                with torch.no_grad():
                    batch_code_repr = normalize(model.code_forward(batch))
                    batch_cmnt_repr = normalize(model.comment_forward(batch))
                code_reprs.append(batch_code_repr)
                cmnt_reprs.append(batch_cmnt_repr)

            code_reprs = torch.cat(code_reprs, dim=0)
            # LOGGER.info(code_reprs.size())
            cmnt_reprs = torch.cat(cmnt_reprs, dim=0)
            # LOGGER.info(cmnt_reprs.size())

            # assert code_reprs.size(0) == cmnt_reprs.size(0)
            data_len = code_reprs.size(0)
            if pool_size == -1:
                pool_size = data_len
                # LOGGER.info(pool_size)

            sim_mat = F.cosine_similarity(code_reprs, cmnt_reprs, dim=-1)
            for ind in range(cmnt_reprs.size(0)):
                cur_cmnt_repr = cmnt_reprs[ind]
                # print(cur_cmnt_repr.size())
                cur_cmnt_repr = cur_cmnt_repr.repeat(pool_size, 1)
                # print(cur_cmnt_repr.size())

                randon_indices = [ind] + random.sample(set(np.setdiff1d(range(sim_mat.size(0)), ind)), pool_size - 1)
                selected_code_reprs = code_reprs.index_select(
                    dim=0,
                    index=torch.Tensor(randon_indices).long().to(code_reprs.device)
                )
                similarity = F.cosine_similarity(cur_cmnt_repr, selected_code_reprs)
                _, predict = similarity.topk(pool_size)
                predict = predict.tolist()
                # print(predict)
                real = [0]  # first selected sim_tensor is our target

                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))

            if pool_size == code_reprs.size(0):
                pool_size = 'all'
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs), pool_size

    @staticmethod
    def summarization_eval(model: IModel, data_loader: DataLoader, token_dicts: TokenDicts, criterion: BaseLoss,
                           collate_func=None, model_filename=None, metrics=None) -> Any:

        with torch.no_grad():
            model.eval()
            data_iter = iter(data_loader)  # init

            total_loss = 0.0
            src_comment_all, trg_comment_all, pred_comment_all, src_code_all = \
                [], [], [], []
            if model.config['training']['pointer']:
                oov_vocab = []
            else:
                oov_vocab = None

            for iteration in range(1, 1 + len(data_loader)):  # 1 + len(data_loader)
                batch = data_iter.__next__()
                if collate_func is not None:
                    batch = collate_func(batch)
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                comment_pred, comment_logprobs, = model.eval_pipeline(batch)
                if model_filename is None:
                    src_comment_all.extend([None] * model.config['training']['batch_size'])
                    src_code_all.extend([None] * model.config['training']['batch_size'])
                else:
                    src_codes, src_comments, = zip(*batch['case_study'])
                    src_comment_all.extend(src_comments)
                    src_code_all.extend(src_codes)

                # comment
                trg_comment_all.extend(batch['comment'][4])
                pred_comment_all.extend(comment_pred)
                # oovs
                if model.config['training']['pointer']:
                    oov_vocab.extend(batch['pointer'][-1])

                # print(comment_logprobs.size())
                # print(comment_target_padded.size())
                if model.config['training']['pointer']:
                    comment_target = batch['pointer'][1][:, :model.config['training']['max_predict_length']]
                else:
                    comment_target = batch['comment'][2][:, :model.config['training']['max_predict_length']]
                # print('comment_logprobs: ', comment_logprobs.size())
                # print('comment_target: ', comment_target.size())
                comment_loss = criterion(comment_logprobs[:, :comment_target.size(1)], comment_target)
                total_loss += comment_loss.item()
            total_loss /= len(data_loader)
            LOGGER.info('Summarization test loss: {:.4}'.format(total_loss))

            if model_filename is None:
                pred_filename = None
            else:
                pred_filename = model_filename.replace('.pt', '.pred')


            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, \
            cider, srcs_return, trgs_return, preds_return, = \
                eval_metrics(src_comment_all, trg_comment_all, pred_comment_all, src_code_all,
                             oov_vocab, token_dicts, pred_filename,
                             metrics=model.config['testing']['metrics'] if metrics is None else metrics, )
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
                map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
                                                            rouge1, rouge2, rouge3, rouge4, rougel, cider,))

            return bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider

    @staticmethod
    def case_study_eval(model: IModel, data_loader: DataLoader, token_dicts: TokenDicts,
                        collate_func=None, model_filename=None, ) -> Any:
        # load ast size
        import glob, ujson
        ast_files = sorted([filename for filename in glob.glob('{}/*'.format(os.path.join(
            model.config['dataset']['dataset_dir'], 'ruby', 'ast'
        ))) if 'test' in filename])

        ast_len = []
        for fl in ast_files:
            with open(fl, 'r') as reader:
                line = reader.readline().strip()
                while len(line) > 0:
                    line = ujson.loads(line)
                    ast_len.append(len(line))
                    line = reader.readline().strip()

        with torch.no_grad():
            model.eval()
            data_iter = iter(data_loader)  # init

            src_comment_all, trg_comment_all, pred_comment_all, src_code_all = \
                [], [], [], []
            tok_len, comment_len = [], []

            if model.config['training']['pointer']:
                oov_vocab = []
            else:
                oov_vocab = None

            for iteration in range(1, 1 + len(data_loader)):  # 1 + len(data_loader)
                batch = data_iter.__next__()
                if collate_func is not None:
                    batch = collate_func(batch)
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                comment_pred, comment_logprobs, = model.eval_pipeline(batch)
                if model_filename is None:
                    src_comment_all.extend([None] * model.config['training']['batch_size'])
                    src_code_all.extend([None] * model.config['training']['batch_size'])
                else:
                    src_codes, src_comments, = zip(*batch['case_study'])
                    src_comment_all.extend(src_comments)
                    src_code_all.extend(src_codes)

                # comment
                trg_comment_all.extend(batch['comment'][4])
                pred_comment_all.extend(comment_pred)
                # oovs
                if model.config['training']['pointer']:
                    oov_vocab.extend(batch['pointer'][-1])

                tok_len.extend(batch['tok'][1].tolist())
                comment_len.extend(batch['comment'][-2].tolist())

            if model_filename is None:
                pred_filename = None
            else:
                pred_filename = model_filename.replace('.pt', '.pred')

            eval_per_metrics(src_comment_all, trg_comment_all, pred_comment_all, src_code_all,
                             oov_vocab, tok_len, comment_len, ast_len,
                             token_dicts, pred_filename, )
            LOGGER.info('write test case-study info in {}'.format(pred_filename))






    @staticmethod
    def case_study_eval_code2seq(model: IModel, data_loader: DataLoader, token_dicts: TokenDicts,
                        collate_func=None, model_filename=None, ) -> Any:
        # load ast size
        # import glob, ujson
        # ast_files = sorted([filename for filename in glob.glob('{}/*'.format(os.path.join(
        #     model.config['dataset']['dataset_dir'], 'ruby', 'ast'
        # ))) if 'test' in filename])
        #
        # ast_len = []
        # for fl in ast_files:
        #     with open(fl, 'r') as reader:
        #         line = reader.readline().strip()
        #         while len(line) > 0:
        #             line = ujson.loads(line)
        #             ast_len.append(len(line))
        #             line = reader.readline().strip()

        ast_len =   load_data(model,datatype='ast')
        tok_len =   load_data(model,datatype='tok')


        with torch.no_grad():
            model.eval()
            data_iter = iter(data_loader)  # init

            src_comment_all, trg_comment_all, pred_comment_all, src_code_all = \
                [], [], [], []
            # tok_len, comment_len = [], []
            comment_len = []

            if model.config['training']['pointer']:
                oov_vocab = []
            else:
                oov_vocab = None

            for iteration in range(1, 1 + len(data_loader)):  # 1 + len(data_loader)
                batch = data_iter.__next__()
                if collate_func is not None:
                    batch = collate_func(batch)
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                comment_pred, comment_logprobs, = model.eval_pipeline(batch)
                if model_filename is None:
                    src_comment_all.extend([None] * model.config['training']['batch_size'])
                    src_code_all.extend([None] * model.config['training']['batch_size'])
                else:
                    src_codes, src_comments, = zip(*batch['case_study'])
                    src_comment_all.extend(src_comments)
                    src_code_all.extend(src_codes)

                # comment
                trg_comment_all.extend(batch['comment'][4])
                pred_comment_all.extend(comment_pred)
                # oovs
                if model.config['training']['pointer']:
                    oov_vocab.extend(batch['pointer'][-1])

                # tok_len.extend(batch['tok'][1].tolist())
                comment_len.extend(batch['comment'][-2].tolist())

            if model_filename is None:
                pred_filename = None
            else:
                pred_filename = model_filename.replace('.pt', '.pred')

            eval_per_metrics(src_comment_all, trg_comment_all, pred_comment_all, src_code_all,
                             oov_vocab, tok_len, comment_len, ast_len,
                             token_dicts, pred_filename, )
            LOGGER.info('write test case-study info in {}'.format(pred_filename))



    @staticmethod
    def summarization_kd_eval(args, model: IModel, dataset, data_loader, token_dicts, criterion, trainer_type,
                              last_best_bleu1_dict=None,
                              train_dataset_names=None, train_dataset_expert_scores=None,
                              epoch=0,
                              best_bleu1=0, best_bleu1_epoch=0, best_cider=0, best_cider_epoch=0,
                              model_filename=None,
                              metrics=METRICS):

        with torch.no_grad():
            model.eval()
            distill = model.config['kd']['distill']
            code_modalities_str = '8'.join(model.config['training']['code_modalities'])
            # index_all = []
            if distill:
                dataset_id_all = []

            data_iter = iter(data_loader)
            total_loss = 0.0

            src_comment_all, trg_comment_all, pred_comment_all, src_code_all, oov_vocab = \
                [], [], [], [], []
            for iteration in range(1, 1 + len(data_loader)):
                batch = data_iter.__next__()
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                # comment_pred, comment_logprobs, comment_target_padded, = model.eval_pipeline(batch)
                comment_pred, comment_logprobs, = model.eval_pipeline(batch)

                if model_filename is None:
                    src_comment_all.extend([None] * model.config['training']['batch_size'])
                    src_code_all.extend([None] * model.config['training']['batch_size'])
                else:
                    src_codes, src_comments, = zip(*batch['case_study'])
                    src_comment_all.extend(src_comments)
                    src_code_all.extend(src_codes)

                # comment
                trg_comment_all.extend(batch['comment'][4])
                pred_comment_all.extend(comment_pred)
                # oovs
                if model.config['training']['pointer']:
                    oov_vocab.extend(batch['pointer'][-1])

                if model.config['training']['pointer']:
                    comment_target = batch['pointer'][1][:, :model.config['training']['max_predict_length']]
                else:
                    comment_target = batch['comment'][2][:, :model.config['training']['max_predict_length']]

                if distill:
                    comment_loss = criterion(comment_logprobs[:, :comment_target.size(1)], comment_target, batch)
                    dataset_id_all.extend(batch['dataset_id'])
                else:
                    comment_loss = criterion(comment_logprobs[:, :comment_target.size(1)], comment_target)

                total_loss += comment_loss.item()
            total_loss /= len(data_loader)
            # LOGGER.info('Summarization test loss: {:.4}'.format(total_loss))

            if model_filename is None:
                pred_filename = None
            else:
                pred_filename = model_filename.replace('.pt', '.pred')

            if distill:
                assert len(set(dataset_id_all)) == dataset.num_dataset, \
                    print(
                        "len(set(dataset_id_all)) :{}  dataset.num_dataset:{} len(dataset_id_all):{} len(dataset):{} dataset_id_all:{} ". \
                            format(len(set(dataset_id_all)), dataset.num_dataset, len(dataset_id_all), len(dataset),
                                   dataset_id_all))
                metric = \
                    calculate_scores_multi_dataset(dataset.num_dataset, dataset_id_all, token_dicts,
                                                   pred_comment_all,
                                                   trg_comment_all, oov_vocab, metrics=metrics)

                bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                    metric['all']['bleu1'], metric['all']['bleu2'], metric['all']['bleu3'], \
                    metric['all']['bleu4'], \
                    metric['all']['meteor'], metric['all']['rouge1'], metric['all']['rouge2'], \
                    metric['all']['rouge3'], metric['all']['rouge4'], metric['all']['rougel'], metric['all'][
                        'cider']

                if args.train_mode == 'test':
                    headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
                    result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                           rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
                    LOGGER.info(
                        'Evaluation results:\n{}'.format(
                            tabulate(result_table, headers=headers, tablefmt='github')))

                # assert bleu1_bak == bleu1

                student_scores = [0 for _ in range(dataset.num_dataset)]
                student_dataset_names = [0 for _ in range(dataset.num_dataset)]
                for ds_id in range(dataset.num_dataset):
                    train_ds_id = train_dataset_names.index(dataset.dataset_names[ds_id])
                    student_scores[train_ds_id] = metric[ds_id]['bleu1']
                    student_dataset_names[train_ds_id] = dataset.dataset_names[ds_id]
                print("student_dataset_names: ", student_dataset_names)
                print("train_dataset_expert_scores: ", train_dataset_expert_scores)
                print("student_scores(bleu1): ", student_scores)

            else:
                student_scores = None
                bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, \
                cider, srcs_return, trgs_return, preds_return, \
                    = eval_metrics(src_comment_all, trg_comment_all, pred_comment_all, src_code_all,
                                   oov_vocab, token_dicts, pred_filename, metrics=metrics)
                bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
                    map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
                                                                rouge1, rouge2, rouge3, rouge4, rougel, cider,))

                if args.train_mode == 'test':
                    headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
                    result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                           rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
                    LOGGER.info(
                        'Evaluation results:\n{}'.format(
                            tabulate(result_table, headers=headers, tablefmt='github')))

            if bleu1 > best_bleu1:
                best_bleu1 = bleu1
                best_bleu1_epoch = epoch

                if not distill:
                    bleu1_dict = {}
                    # name = "bleu1_{}_{}".format(model.config['kd']['source'], model.config['kd']['target'])
                    name = "bleu1_{}_{}".format(model.config['kd']['source'][0], model.config['kd']['target'][0])
                    bleu1_dict[name] = bleu1
                    json_path = os.path.join(model.config['kd']['kd_path'],
                                             'expert_bleu1_{}_{}_{}_{}_epoch{}.json'.format(trainer_type,
                                                                                            model.config['kd'][
                                                                                                'source'][0],
                                                                                            model.config['kd'][
                                                                                                'target'][0],
                                                                                            code_modalities_str,
                                                                                            epoch))
                    last_best_bleu1_dict = bleu1_dict
                    save_json(bleu1_dict, json_path)
            else:
                if not distill:
                    json_path = os.path.join(model.config['kd']['kd_path'],
                                             'expert_bleu1_{}_{}_{}_{}_epoch{}.json'.format(trainer_type,
                                                                                            model.config['kd'][
                                                                                                'source'][0],
                                                                                            model.config['kd'][
                                                                                                'target'][0],
                                                                                            code_modalities_str,
                                                                                            epoch))
                    save_json(last_best_bleu1_dict, json_path)

            if cider > best_cider:
                best_cider = cider
                best_cider_epoch = epoch

            LOGGER.info("Epoch: {} val_metric bleu1:{} cider:{} loss:{} ".format(epoch, bleu1, cider, total_loss))
            LOGGER.info("Epoch: {} best_bleu1: {} best_bleu1_epoch:{} best_cider:{} best_cider_epoch:{} ".format(
                epoch, best_bleu1, best_bleu1_epoch, best_cider, best_cider_epoch))

            return student_scores, best_bleu1, best_bleu1_epoch, best_cider, best_cider_epoch, last_best_bleu1_dict
        # @staticmethod
        # def summarization_eval_return(model: IModel, data_loader: DataLoader, token_dicts: TokenDicts, criterion: BaseLoss,
        #                               model_filename=None, metrics=METRICS) -> Any:
        #
        #     with torch.no_grad():
        #         model.eval()
        #         data_iter = iter(data_loader)  # init
        #
        #         total_loss = 0.0
        #         src_comment_all, trg_comment_all, pred_comment_all, src_code_all = \
        #             [], [], [], []
        #         if model.config['training']['pointer']:
        #             oov_vocab = []
        #         else:
        #             oov_vocab = None
        #
        #         for iteration in range(1, 1 + len(data_loader)):  # 1 + len(data_loader)
        #             batch = data_iter.__next__()
        #             if model.config['common']['device'] is not None:
        #                 batch = batch_to_cuda(batch)
        #             comment_pred, comment_logprobs, = model.eval_pipeline(batch)
        #             if model_filename is None:
        #                 src_comment_all.extend([None] * model.config['training']['batch_size'])
        #                 src_code_all.extend([None] * model.config['training']['batch_size'])
        #             else:
        #                 # only runs for valid
        #                 src_comments, src_codes, = zip(*batch['case_study'])
        #                 src_comment_all.extend(src_comments)
        #                 src_code_all.extend(src_codes)
        #
        #             # comment
        #             trg_comment_all.extend(batch['comment'][4])
        #             pred_comment_all.extend(comment_pred)
        #             # oovs
        #             if model.config['training']['pointer']:
        #                 oov_vocab.extend(batch['pointer'][-1])
        #
        #             # print(comment_logprobs.size())
        #             # print(comment_target_padded.size())
        #             if model.config['training']['pointer']:
        #                 comment_target = batch['pointer'][1][:, :model.config['training']['max_predict_length']]
        #             else:
        #                 comment_target = batch['comment'][2][:, :model.config['training']['max_predict_length']]
        #             # print('comment_logprobs: ', comment_logprobs.size())
        #             # print('comment_target: ', comment_target.size())
        #             comment_loss = criterion(comment_logprobs[:, :comment_target.size(1)], comment_target)
        #             total_loss += comment_loss.item()
        #         total_loss /= len(data_loader)
        #         LOGGER.info('Summarization test loss: {:.4}'.format(total_loss))
        #
        #         if model_filename is None:
        #             pred_filename = None
        #         else:
        #             pred_filename = model_filename.replace('.pt', '.pred')
        #
        #         bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, \
        #         cider, srcs_return, trgs_return, preds_return, = \
        #             eval_metrics(src_comment_all, trg_comment_all, pred_comment_all, src_code_all,
        #                          oov_vocab, token_dicts, pred_filename, metrics=metrics, )
        #         bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
        #             map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
        #                                                         rouge1, rouge2, rouge3, rouge4, rougel, cider,))
        #         # LOGGER.info('B1: {:.4f}, B2: {:.4f}, B3: {:.4f}, B4: {:.4f}, Meteor: {:.4f}, '
        #         #             'R1: {:.4f}, R2: {:.4f}, R3: {:.4f}, R4: {:.4f}, RL: {:.4f}, Cider: {:.4f}'. \
        #         #             format(bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, ))
        #
        #         # headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
        #         # result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
        #         #                                        rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
        #         # LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers, tablefmt='github')))
        #         return bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider


        # @staticmethod
        # def summarization_eval_return(model: IModel, data_loader: DataLoader, token_dicts: TokenDicts, criterion: BaseLoss,
        #                               model_filename=None, metrics=METRICS) -> Any:
        #
        #     with torch.no_grad():
        #         model.eval()
        #         data_iter = iter(data_loader)  # init
        #
        #         total_loss = 0.0
        #         src_comment_all, trg_comment_all, pred_comment_all, src_code_all = \
        #             [], [], [], []
        #         if model.config['training']['pointer']:
        #             oov_vocab = []
        #         else:
        #             oov_vocab = None
        #
        #         for iteration in range(1, 1 + len(data_loader)):  # 1 + len(data_loader)
        #             batch = data_iter.__next__()
        #             if model.config['common']['device'] is not None:
        #                 batch = batch_to_cuda(batch)
        #             comment_pred, comment_logprobs, = model.eval_pipeline(batch)
        #             if model_filename is None:
        #                 src_comment_all.extend([None] * model.config['training']['batch_size'])
        #                 src_code_all.extend([None] * model.config['training']['batch_size'])
        #             else:
        #                 # only runs for valid
        #                 src_comments, src_codes, = zip(*batch['case_study'])
        #                 src_comment_all.extend(src_comments)
        #                 src_code_all.extend(src_codes)
        #
        #             # comment
        #             trg_comment_all.extend(batch['comment'][4])
        #             pred_comment_all.extend(comment_pred)
        #             # oovs
        #             if model.config['training']['pointer']:
        #                 oov_vocab.extend(batch['pointer'][-1])
        #
        #             # print(comment_logprobs.size())
        #             # print(comment_target_padded.size())
        #             if model.config['training']['pointer']:
        #                 comment_target = batch['pointer'][1][:, :model.config['training']['max_predict_length']]
        #             else:
        #                 comment_target = batch['comment'][2][:, :model.config['training']['max_predict_length']]
        #             # print('comment_logprobs: ', comment_logprobs.size())
        #             # print('comment_target: ', comment_target.size())
        #             comment_loss = criterion(comment_logprobs[:, :comment_target.size(1)], comment_target)
        #             total_loss += comment_loss.item()
        #         total_loss /= len(data_loader)
        #         LOGGER.info('Summarization test loss: {:.4}'.format(total_loss))
        #
        #         if model_filename is None:
        #             pred_filename = None
        #         else:
        #             pred_filename = model_filename.replace('.pt', '.pred')
        #
        #         bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, \
        #         cider, srcs_return, trgs_return, preds_return, = \
        #             eval_metrics(src_comment_all, trg_comment_all, pred_comment_all, src_code_all,
        #                          oov_vocab, token_dicts, pred_filename, metrics=metrics, )
        #         bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
        #             map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
        #                                                         rouge1, rouge2, rouge3, rouge4, rougel, cider,))
        #         # LOGGER.info('B1: {:.4f}, B2: {:.4f}, B3: {:.4f}, B4: {:.4f}, Meteor: {:.4f}, '
        #         #             'R1: {:.4f}, R2: {:.4f}, R3: {:.4f}, R4: {:.4f}, RL: {:.4f}, Cider: {:.4f}'. \
        #         #             format(bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, ))
        #
        #         # headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
        #         # result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
        #         #                                        rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
        #         # LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers, tablefmt='github')))
        #         return bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider