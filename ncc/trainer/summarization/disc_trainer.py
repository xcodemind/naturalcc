# -*- coding: utf-8 -*-
import os
import datetime
import ujson
import time
from collections import OrderedDict
import torch
from torch.optim.optimizer import Optimizer
from ncc import LOGGER
from ncc.trainer.trainer_ import Trainer
from ncc.models.template import IModel
from ncc.dataset import UnilangDataloader
from ncc.metric import BaseLoss
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.utils import clean_up_sentence, indices_to_words
from typing import Dict


class DiscTrainer(Trainer):
    '''
    Discriminator Trainer
    '''

    def __init__(self, args: Dict, ) -> None:
        super(DiscTrainer, self).__init__(args)

    def generate_training_pairs(self, model: IModel, dataset: UnilangDataloader, filepath: str):
        model.eval()
        train_data_iter = iter(dataset['train'])
        training_pairs, labels = [], []
        LOGGER.debug('filepath: {}'.format(filepath))
        with open(filepath, 'w') as writer:
            predictions_dict = {}
            for iteration in range(1, 1 + len(dataset['train'])):  # 1 + len(dataset['train'])
                batch = train_data_iter.__next__()
                if model.args['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
                batch_size = comment.size(0)

                enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
                sample_opt = {'sample_max': 1, 'seq_length': model.args['training']['max_predict_length']}
                # seq, seq_logprobs, seq_logp_gathered, seq_padding_mask, seq_lprob_sum, dec_output, dec_hidden, \
                #     = model.decoder.forward(batch, enc_output, dec_hidden, enc_mask, sample_opt)
                seq, seq_logprobs, seq_logp_gathered, seq_lprob_sum, comment_target_padded, = \
                    model.decoder.sample(batch, enc_output, dec_hidden, enc_mask, sample_opt)

                # print('seq: ', seq.size())
                # print(seq)
                # print('batch: ', len(batch['index']))
                # print(batch['index'])
                # seq = seq.tolist()
                oov_vocab = batch['pointer'][-1]
                for i in range(seq.size(0)):
                    # pred = clean_up_sentence(seq[i], remove_UNK=False, remove_EOS=True)
                    # pred = id2word(pred, dict_comment, oov_vocab[i])
                    # print('seq[i]: ', seq[i].size())
                    # print(seq[i])
                    pred = clean_up_sentence(seq[i], remove_EOS=True)
                    pred = indices_to_words(pred, dataset.token_dicts['comment'], oov_vocab[i])
                    # print('pred: ', pred)
                    predictions_dict[batch['index'][i].item()] = pred
            # print('predictions_dict: ', predictions_dict)
            predictions_dict_sorted = dict(OrderedDict(sorted(predictions_dict.items(), key=lambda x: x[0])))
            # print('predictions_dict_sorted: ', predictions_dict_sorted)
            assert len(predictions_dict_sorted) == dataset.size['train']
            for ind in len(predictions_dict_sorted):
                # pred: ['match', 'the', 'conditions', '.']
                writer.write(ujson.dumps(predictions_dict_sorted[ind]) + '\n')
            LOGGER.info('Save file to {}.'.format(filepath))
            # f.writelines()
            # assert False
            # real_pairs = list(zip(batch, comment))  # [(query_seq, response_seq)] * batch_size
            # fake_pairs = list(zip(batch, seq))  # [(query_seq, out_seq)] * batch_size
            # training_pairs.extend(real_pairs)
            # labels.extend([1] * batch_size)
            # training_pairs.extend(fake_pairs)
            # labels.extend([0] * batch_size)
        # 1. 存batch raw code ＝》disc_dataloader
        # 2. batch
        # 64x35
        # code, ..., comment, 0/1
        #
        # 64x27

    def train(self, disc: IModel, dataset: UnilangDataloader, criterion: BaseLoss, disc_optimizer: Optimizer,
              SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time

        for epoch in range(1, 1 + disc.args['gan']['train_epoch_disc']):
            disc.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if disc.args['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                disc_loss = disc.train_sl(batch, criterion)
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, disc_loss.item()))
                disc_optimizer.zero_grad()
                disc_loss.backward()
                total_loss += disc_loss.item()
                disc_optimizer.step()

                if iteration % disc.args['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; time: {}'.format(
                        epoch, disc.args['gan']['train_epoch_critic'], iteration, len(dataset['train']),
                        total_loss / iteration,
                        str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            if epoch <= disc.args['gan']['train_epoch_disc']:
                if SAVE_DIR is not None:
                    model_name = 'disc-{}-bs{}-lr{}-attn{}-pointer{}-ep{}'.format(
                        '8'.join(disc.args['training']['code_modalities']),
                        disc.args['training']['batch_size'],
                        disc.args['gan']['lr_disc'],
                        disc.args['training']['attn_type'],
                        disc.args['training']['pointer'], epoch)
                    model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                    torch.save(disc.state_dict(), model_path)
                    LOGGER.info('Dumping disc in {}'.format(model_path))
                # Evaluator.summarization_eval(critic, dataset['valid'], dataset.token_dicts, )
            else:
                pass
        LOGGER.info('{} train end'.format(self))
