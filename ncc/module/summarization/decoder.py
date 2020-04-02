# -*- coding: utf-8 -*-
import numpy as np
from ncc import LOGGER
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.module.code2vec.base import Encoder_Emb, Encoder_RNN
from ncc.module.attention import GlobalAttention, IntraAttention, SelfAttention
from ncc.data import TokenDicts
from ncc.utils.utils import indices_to_words, clean_up_sentence
from eval.summarization import Bleu, Cider, Rouge
from ncc.utils.constants import *
from typing import Dict, List, Any, Tuple


class SeqDecoder(nn.Module):
    def __init__(self, token_num: int, embed_size: int,
                 rnn_type: str, hidden_size: int, layer_num: int, dropout: float, bidirectional: bool,
                 attn_type: str, pointer: bool, max_predict_length: int, code_modalities: List,
                 decoder_input_feed=False) -> None:
        super(SeqDecoder, self).__init__()
        # embedding params
        self.token_num = token_num
        self.embed_size = embed_size
        # rnn params
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.attn_type = attn_type  # attention mech
        self.pointer = pointer  # pointer-generator
        self.max_predict_length = max_predict_length  # decoder predict length
        # code_modalities
        if 'sbt' in code_modalities:
            code_modalities[code_modalities.index('sbt')] = 'tok'
        # LOGGER.info(code_modalities)
        self.code_modalities = code_modalities
        self.decoder_input_feed = decoder_input_feed
        # net
        self.wemb = Encoder_Emb(self.token_num, self.embed_size, )
        self.rnn = Encoder_RNN(self.rnn_type, self.embed_size, self.hidden_size, self.layer_num, dropout, bidirectional)
        self.concat_map = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.generate_linear = nn.Linear(self.hidden_size, self.token_num)
        if len(self.code_modalities) > 1:
            self.fuse_linear = nn.Linear(self.hidden_size * len(self.code_modalities), self.hidden_size)
        if decoder_input_feed:
            self.decoder_input_linear = nn.Linear(self.hidden_size + self.embed_size, self.embed_size)
        # dropout
        self.dropout = dropout
        # attention
        if self.attn_type is None:
            self.attns = None
            if self.pointer:
                raise NotImplementedError('No pointer when attention is None')
        elif self.attn_type in ['general', 'dot', 'mlp']:
            self.attns = nn.ModuleList(
                [GlobalAttention(hidden_size, coverage=False, attn_type=attn_type)] * len(self.code_modalities))
            if self.pointer:
                self.p_gen_linear = nn.Linear(self.hidden_size * 2 + self.embed_size, 1)
        elif self.attn_type == 'intra':
            self.attns = nn.ModuleList([IntraAttention(hidden_size, self.token_num)] * len(self.code_modalities))
            if self.pointer:
                self.p_gen_linear = nn.Linear(self.hidden_size * 3 + self.embed_size, 1)
        elif self.attn_type == 'self':
            self.attns = SelfAttention(hidden_size, self.token_num)
        else:
            raise NotImplementedError('No such attention({}).'.format(self.attn_type))

    @classmethod
    def load_from_config(cls, config: Dict, modal: str, ) -> Any:
        instance = cls(
            token_num=config['training']['token_num'][modal],
            embed_size=config['training']['embed_size'],
            rnn_type=config['training']['rnn_type'],
            hidden_size=config['training']['rnn_hidden_size'],
            layer_num=config['training']['rnn_layer_num'],
            dropout=config['training']['dropout'],
            bidirectional=False,
            attn_type=config['training']['attn_type'],
            pointer=config['training']['pointer'],
            max_predict_length=config['training']['max_predict_length'],
            code_modalities=config['training']['code_modalities'],
            decoder_input_feed=config['training']['decoder_input_feed'],
        )
        return instance

    def init_hidden(self, batch_size: int) -> Any:
        return self.rnn.init_hidden(batch_size)

    def forward(self, batch, enc_output, dec_hidden, enc_padding_mask, sample_opt={}) -> Tuple:
        sample_max, seq_length = sample_opt.get('sample_max', 1), \
                                 sample_opt.get('seq_length', self.max_predict_length)
        # print('batch...')
        # pprint(batch)
        # (batch_size*mLen, batch_size*(mLen+1),)
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        device = comment.device
        batch_size = comment.size(0)
        seq_length = min(comment.size(1) + 1, seq_length)  # +1 is for EOS
        input = torch.zeros(batch_size, 1).long().fill_(BOS).to(device)  # (batch_size*1) and all items are 2 (BOS)
        # Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise
        mask = torch.LongTensor(batch_size).fill_(1).to(device)  # (batch_size) and all items are 1
        seq, seq_logp_gathered, seq_lprob_sum = torch.zeros(batch_size, seq_length).long().to(device), \
                                                torch.zeros(batch_size, seq_length).to(device), \
                                                torch.zeros(batch_size, seq_length).to(device)
        # For attention
        if self.attn_type == 'intra':  # attention and pointer initialization
            sum_temporal_srcs = {modal: None for modal in self.code_modalities}
            prev_s = {modal: None for modal in self.code_modalities}
        elif self.attn_type is None:
            LOGGER.debug('no attention in {}'.format(self.__class__.__name__))

        # For pointer
        if self.pointer:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
            seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num + pointer_extra_zeros.size(1)).to(device)
        else:
            seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)

        seq_padding_mask, dec_output = [], []
        output_attns_final = torch.zeros((batch_size, self.hidden_size)).to(
            device)  # initialized context vector for decoder_input_feed
        for t in range(seq_length):
            # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            use_ground_truth = (torch.rand(batch_size) > 0.25).unsqueeze(1).long().to(device)
            # Select decoder input based on use_ground_truth probabilities
            input = use_ground_truth * comment_input[:, t].unsqueeze(1) + (1 - use_ground_truth) * input

            if self.attn_type is None:
                input_emb = self.wemb(input)  # (batch_size*1*emb_size)
                if self.decoder_input_feed:
                    input_emb = self.decoder_input_linear(
                        torch.cat([input_emb.squeeze(1), dec_hidden[0][-1]], dim=1). \
                            unsqueeze(1))  # the initial output_attns_final is torch.zeros
                output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
                dec_output.append(output.squeeze(1))  # one step decode (batch_size*rnn_hidden_size, )
                final_out = dec_hidden[0][-1]
                final_out = F.dropout(final_out, self.dropout, training=self.training)
                decoded = self.generate_linear(final_out)  # (batch_size*comment_dict_size)
                logprobs = F.log_softmax(decoded, dim=-1)  # (batch_size*comment_dict_size)
                prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)
            elif self.attn_type in ['general', 'dot', 'mlp']:
                input_emb = self.wemb(input)  # (batch_size*1*emb_size)
                if self.decoder_input_feed:
                    input_emb = self.decoder_input_linear(torch.cat([input_emb.squeeze(1), output_attns_final],
                                                                    dim=1).unsqueeze(
                        1))  # the initial output_attns_final is torch.zeros
                output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
                dec_output.append(output.squeeze(1))  # one step decode (batch_size*rnn_hidden_size, )
                # output = output.squeeze(1)  # one step decode (batch_size*rnn_hidden_size, )
                output_attns = {k: None for k in enc_output.keys()}
                p_attns = {k: None for k in enc_output.keys()}
                for i, modal in enumerate(self.code_modalities):
                    # (batch_size*rnn_hidden_size, batch_size*code_len), dec_hidden[0][-1], 0: h, -1: last layer
                    output_attns[modal], p_attns[modal] = self.attns[i](dec_hidden[0][-1], enc_output[modal],
                                                                        enc_padding_mask[modal])
                # fuse multi-modality
                if len(self.code_modalities) > 1:
                    output_attns_final = torch.tanh(self.fuse_linear(
                        torch.cat([output_attns[key] for key in sorted(output_attns.keys())],
                                  dim=1)))  # batch_size*rnn_hidden_size
                else:  # single modality
                    output_attns_final = output_attns[self.code_modalities[0]]
                LOGGER.debug('output_attns_final: {}'.format(output_attns_final.size()))
                # concatenate context vector and hidden state
                final_out = torch.tanh(self.concat_map(
                    torch.cat((output_attns_final, dec_hidden[0][-1]), dim=1)))  # batch_size*rnn_hidden_size
                LOGGER.debug('final_out: {}'.format(final_out.size()))
                final_out = F.dropout(final_out, self.dropout, training=self.training)
                decoded = self.generate_linear(final_out)  # (batch_size*comment_dict_size)
                logprobs = F.log_softmax(decoded, dim=1)  # (batch_size*comment_dict_size)
                prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)
                if self.pointer:
                    st_hat = dec_hidden[0][-1]  # (batch_size*rnn_hidden_size)
                    p_gen_input = torch.cat([output_attns['tok'], st_hat, input_emb.squeeze(1)],
                                            1)  # B x (2*2*hidden_dim + emb_dim) only copy from token modality
                    p_gen = self.p_gen_linear(p_gen_input)
                    p_gen = torch.sigmoid(p_gen)
                    prob_prev_ = p_gen * prob_prev
                    p_attn_ = (1 - p_gen) * p_attns['tok']  # only copy from token modality
                    if pointer_extra_zeros is not None:
                        prob_prev_ = torch.cat([prob_prev_, pointer_extra_zeros], 1)

                    prob_prev = prob_prev_.scatter_add(1, code_dict_comment, p_attn_)
                    logprobs = torch.log(prob_prev + EPS_ZERO)  # self.opt.eps,  + 1e-12

            elif self.attn_type == 'intra':
                input_emb = self.wemb(input)  # (batch_size*1*emb_size)
                if self.decoder_input_feed:
                    input_emb = self.decoder_input_linear(torch.cat([input_emb.squeeze(1), output_attns_final],
                                                                    dim=1).unsqueeze(
                        1))  # the initial output_attns_final is torch.zeros
                output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
                dec_output.append(output.squeeze(1))  # one step decode (batch_size*rnn_hidden_size, )
                output_attns = {k: None for k in self.code_modalities}
                ct_e = {k: None for k in self.code_modalities}
                ct_d = {k: None for k in self.code_modalities}
                st_hat = {k: None for k in self.code_modalities}
                p_attns = {k: None for k in self.code_modalities}

                for i, modal in enumerate(self.code_modalities):
                    # output_attn: batch_size*rnn_hidden_size
                    # ct_e: batch_size*rnn_hidden_size
                    # ct_d: batch_size*rnn_hidden_size
                    # st_hat: batch_size*rnn_hidden_size
                    # p_attn: batch_size*code_len
                    output_attns[modal], sum_temporal_srcs[modal], prev_s[modal], ct_e[modal], ct_d[modal], st_hat[
                        modal], p_attns[modal] = \
                        self.attns[i](input_emb, enc_output[modal], enc_padding_mask[modal],
                                      sum_temporal_srcs[modal], dec_hidden, prev_s[modal])
                # fuse multi-modality
                if len(self.code_modalities) > 1:
                    output_attns_final = self.fuse_linear(
                        torch.cat([output_attns[key] for key in sorted(output_attns.keys())],
                                  dim=1))  # batch_size*rnn_hidden_size
                else:  # single modality
                    output_attns_final = output_attns[self.code_modalities[0]]
                # concatenate context vector and hidden state
                final_out = torch.tanh(self.concat_map(
                    torch.cat((output_attns_final, dec_hidden[0][-1]), dim=1)))  # batch_size*rnn_hidden_size
                final_out = F.dropout(final_out, self.dropout, training=self.training)
                decoded = self.generate_linear(final_out)  # (batch_size*comment_dict_size)
                logprobs = F.log_softmax(decoded, dim=1)  # (batch_size*comment_dict_size)
                prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)

                if self.pointer:
                    p_gen = torch.cat([ct_e['tok'], ct_d['tok'], st_hat['tok'], input_emb.squeeze(1)], 1)
                    p_gen = self.p_gen_linear(p_gen)
                    p_gen = torch.sigmoid(p_gen)
                    prob_prev = p_gen * prob_prev
                    p_attn_ = (1 - p_gen) * p_attns['tok']
                    if pointer_extra_zeros is not None:
                        prob_prev = torch.cat([prob_prev, pointer_extra_zeros], dim=1)
                    prob_prev = prob_prev.scatter_add(1, code_dict_comment, p_attn_)
                    logprobs = torch.log(prob_prev + EPS_ZERO)  # self.opt.eps,  + 1e-12
            else:
                raise NotImplementedError

            if sample_max:
                sample_logprobs, predicted = torch.max(logprobs, 1)
                seq[:, t] = predicted.reshape(-1)
                seq_logp_gathered[:, t] = sample_logprobs
                seq_logprobs[:, t, :] = logprobs
            else:
                predicted = torch.multinomial(prob_prev, 1)  # .to(device)
                seq[:, t] = predicted.reshape(-1)
                seq_logp_gathered[:, t] = logprobs.gather(1, predicted).reshape(-1)
                seq_logprobs[:, t, :] = logprobs

            seq_lprob_sum[:, t] = logprobs.sum(dim=-1)
            input = predicted.reshape(-1, 1)
            # .detach() Mask indicating whether sampled word is OOV
            is_oov = (input >= self.token_num).long().to(device).detach()
            LOGGER.debug('is_oov: {}'.format(is_oov.size()))
            input = (1 - is_oov) * input + (is_oov) * UNK

            mask_t = torch.zeros(batch_size).to(device)  # Padding mask of batch for current time step
            mask_t[mask == 1] = 1  # If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            # If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            mask[(mask == 1).int() + (input.squeeze(1) == EOS).int() == 2] = 0
            seq_padding_mask.append(mask_t)

        seq_padding_mask = torch.stack(seq_padding_mask).t()  # (batch_size*max_len)
        dec_output = torch.stack(dec_output).transpose(0, 1)  # (batch_size*max_len*rnn_hidden_size)

        return seq, seq_logprobs, seq_logp_gathered, seq_padding_mask, seq_lprob_sum, dec_output, dec_hidden,

    def forward_pg(self, batch, enc_output, dec_hidden, enc_padding_mask, token_dicts: TokenDicts, sample_opt={},
                   reward_func='bleu', oov_vocab=None) -> Tuple:
        sample_max, seq_length = sample_opt.get('sample_max', 1), \
                                 sample_opt.get('seq_length', self.max_predict_length)
        # print('batch...')
        # pprint(batch)
        # (batch_size*mLen, batch_size*(mLen+1),)
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        device = comment.device
        batch_size = comment.size(0)
        seq_length = min(comment.size(1) + 1, seq_length)  # +1 is for EOS

        seq, seq_logprobs, seq_logp_gathered, seq_padding_mask, seq_lprob_sum, dec_output, dec_hidden, = \
            self.forward(batch, enc_output, dec_hidden, enc_padding_mask, sample_opt)
        seq_list = seq.tolist()
        rewards = []
        for idx in range(seq.size(0)):
            if oov_vocab:
                pred = indices_to_words(clean_up_sentence(seq[idx], remove_EOS=True),
                                        token_dicts['comment'], oov_vocab[idx])
            else:
                pred = indices_to_words(clean_up_sentence(seq[idx], remove_EOS=True),
                                        token_dicts['comment'], oov_vocab=None)
            # 50001 getLabel should not be UNK_WORD, should be copied from code.so use id2word
            res = {0: [' '.join(pred)]}
            gts = {0: [' '.join(raw_comment[idx])]}
            if reward_func == 'bleu':
                score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
                rewards.append(np.mean(scores_Bleu[0]))
            elif reward_func == 'cider':  # TODO: exist bug, reward is always 0
                print('gts: ', gts)
                print('res: ', res)
                score_Cider, scores_Cider = Cider().compute_score(gts, res)
                print('score_Cider: ', score_Cider)
                print('scores_Cider: ', scores_Cider)
                rewards.append(score_Cider)
            elif reward_func == 'rouge':
                score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
                rewards.append(score_Rouge)

        rewards = torch.Tensor(rewards).reshape(-1, 1).repeat(1, seq_length).to(device)
        LOGGER.debug('rewards: {}'.format(rewards.size()))

        return seq, seq_logprobs, seq_logp_gathered, seq_padding_mask, rewards, seq_lprob_sum, dec_output, dec_hidden,

    # TODO
    def forward_gan(self, disc, code_batch, code_length, comment_batch, dec_hidden, sample_opt={}):
        sample_max, seq_length = sample_opt.get('sample_max', 1), sample_opt.get('seq_length',
                                                                                 self.opt.max_predict_length)
        batch_size = dec_hidden[0].size(1)
        input = torch.zeros(1, batch_size).long().fill_(BOS)
        input = to_cuda(self.opt, input)
        mask = torch.LongTensor(batch_size).fill_(
            1).cuda()  # Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise

        seq, seq_logprobs, seq_logp_gathered, rewards = torch.zeros(seq_length, batch_size).long().cuda(), \
                                                        torch.zeros(seq_length, batch_size,
                                                                    self.dict_comment.size()).cuda(), torch.zeros(
            seq_length,
            batch_size).cuda(), \
                                                        torch.zeros(seq_length, batch_size).cuda()
        seq_padding_mask = []

        for t in range(seq_length):
            input_emb = self.wemb(input)
            output, dec_hidden = self.rnn(input_emb, dec_hidden)
            # output = F.dropout(output, self.dropout, training=self.training)
            decoded = self.generate_linear(output.reshape(-1, output.size(2)))
            logprobs = F.log_softmax(decoded, dim=1)  # self.beta *    logprobs==>logprob
            prob_prev = torch.exp(logprobs)  # .cpu() # fetch prev distribution: shape Nx(M+1)

            if sample_max:
                sample_logprobs, predicted = torch.max(logprobs, 1)
                seq[t, :] = predicted.reshape(-1)
                seq_logp_gathered[t, :] = sample_logprobs
                seq_logprobs[t, :, :] = logprobs

            else:
                predicted = torch.multinomial(prob_prev, 1)  # .cuda()
                seq[t, :] = predicted.reshape(-1)
                seq_logp_gathered[t, :] = logprobs.gather(1, predicted).reshape(-1)
                seq_logprobs[t, :, :] = logprobs

            input = predicted.reshape(1, -1)
            mask_t = torch.zeros(batch_size).cuda()  # Padding mask of batch for current time step
            # print('forward_pg-mask_t: ', mask_t.type(), mask_t.size())
            # print(mask_t)
            mask_t[
                mask == 1] = 1  # If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0

            mask[(mask == 1).int() + (
                    input.squeeze() == EOS).int() == 2] = 0  # If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            # print('forward_pg-mask-: ', mask.type(), mask.size())
            # print(mask)
            seq_padding_mask.append(mask_t)
            # is_oov = (input >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
            # input = (1 - is_oov) * input + (is_oov) * UNK  # Replace OOVs with [UNK] token

        seq_padding_mask = torch.stack(seq_padding_mask).long()
        comment_fake = seq * seq_padding_mask

        with torch.no_grad():
            reward = disc(comment_fake)
            reward = torch.exp(reward[:, 1])
            rewards = reward.repeat(seq_length, 1)

        return seq, seq_logprobs, seq_logp_gathered, seq_padding_mask, rewards, dec_hidden

    def sample(self, batch, enc_output, dec_hidden, enc_padding_mask, sample_opt={}) -> Tuple:
        sample_max, beam_size, seq_length = sample_opt.get('sample_max', 1), \
                                            sample_opt.get('beam_size', 1), \
                                            sample_opt.get('seq_length', self.max_predict_length)
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        # pprint(batch)
        device = comment.device
        batch_size = comment.size(0)

        # batch_size = dec_hidden[0].size(1)
        # device = batch['comment'][0].device  # cpu or gpu
        input = torch.zeros(batch_size, 1).long().fill_(BOS).to(device)

        if beam_size > 1:
            return self.sample_beam(input, dec_hidden)

        seq, seq_logp_gathered, seq_lprob_sum = torch.zeros(batch_size, seq_length).long().to(device), \
                                                torch.zeros(batch_size, seq_length).to(device), \
                                                torch.zeros(batch_size, seq_length).to(device)

        # comment_target = batch['pointer'][1] if self.pointer else batch['comment'][2]
        # print(comment_target.size())
        # if comment_target.shape[1] < seq_length:
        #     comment_target_padded = torch.zeros(batch_size, seq_length).long().to(device)
        #     comment_target_padded[:, :comment_target.shape[1]] = comment_target
        # else:
        #     # it may longer than seq_length
        #     # print('comment_target: ', comment_target.size())
        #     comment_target_padded = comment_target

        # For attention
        if self.attn_type:  # only attention in ['general', 'dot', 'mlp', 'intra'], no pointer initialization
            if isinstance(self.code_modalities, str):  # for uni modality
                enc_output = enc_output[self.code_modalities]  # (batch_size*m_len*rnn_hidden_size)
                enc_padding_mask = enc_padding_mask[self.code_modalities]  # (batch_size*m_len): 0 or 1
            elif isinstance(self.code_modalities, list):  # for multi modalities
                pass
        if self.attn_type == 'intra':  # attention and pointer initialization
            sum_temporal_srcs = None
        elif self.attn_type is None:
            LOGGER.debug('no attention in {}'.format(self.__class__.__name__))
            pass

        # For pointer
        if self.pointer:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
            seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num + pointer_extra_zeros.size(1)).to(
                device)
        else:
            seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)

        prev_s = None
        for t in range(seq_length):
            if self.attn_type is None:
                input_emb = self.wemb(input)  # (batch_size*1*emb_size)
                output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
                final_out = dec_hidden[0][-1]
                final_out = F.dropout(final_out, self.dropout, training=self.training)
                decoded = self.generate_linear(final_out)  # (batch_size*comment_dict_size)
                logprobs = F.log_softmax(decoded, dim=-1)  # (batch_size*comment_dict_size)
                prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)
            elif self.attn_type in ['general', 'dot', 'mlp']:
                input_emb = self.wemb(input)  # (batch_size*1*emb_size)
                output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
                output_attns = {k: None for k in enc_output.keys()}
                p_attns = {k: None for k in enc_output.keys()}
                for i, modal in enumerate(self.code_modalities):
                    # (batch_size*rnn_hidden_size, batch_size*code_len), dec_hidden[0][-1], 0: h, -1: last layer
                    output_attns[modal], p_attns[modal] = self.attns[i](dec_hidden[0][-1], enc_output[modal],
                                                                        enc_padding_mask[modal])
                # fuse multi-modality
                if len(self.code_modalities) > 1:
                    output_attns_final = torch.tanh(self.fuse_linear(
                        torch.cat(
                            [output_attns[key] for key in sorted(output_attns.keys())],
                            dim=1)))  # batch_size*rnn_hidden_size
                else:  # single modality
                    output_attns_final = output_attns[self.code_modalities[0]]
                LOGGER.debug('output_attns_final: {}'.format(output_attns_final.size()))
                # concatenate context vector and hidden state
                final_out = torch.tanh(self.concat_map(
                    torch.cat((output_attns_final, dec_hidden[0][-1]), dim=1)))  # batch_size*rnn_hidden_size
                LOGGER.debug('final_out: {}'.format(final_out.size()))
                final_out = F.dropout(final_out, self.dropout, training=self.training)
                decoded = self.generate_linear(final_out)  # (batch_size*comment_dict_size)
                logprobs = F.log_softmax(decoded, dim=1)  # (batch_size*comment_dict_size)
                prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)
                if self.pointer:
                    st_hat = dec_hidden[0][-1]  # (batch_size*rnn_hidden_size)
                    p_gen_input = torch.cat([output_attns['tok'], st_hat, input_emb.squeeze(1)],
                                            1)  # B x (2*2*hidden_dim + emb_dim) only copy from token modality
                    p_gen = self.p_gen_linear(p_gen_input)
                    p_gen = torch.sigmoid(p_gen)
                    prob_prev_ = p_gen * prob_prev
                    p_attn_ = (1 - p_gen) * p_attns['tok']  # only copy from token modality
                    if pointer_extra_zeros is not None:
                        prob_prev_ = torch.cat([prob_prev_, pointer_extra_zeros], 1)

                    prob_prev = prob_prev_.scatter_add(1, code_dict_comment, p_attn_)
                    logprobs = torch.log(prob_prev + EPS_ZERO)  # self.opt.eps,  + 1e-12

            elif self.attn_type == 'intra':
                input_emb = self.wemb(input)  # (batch_size*1*emb_size)
                output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )

                output_attns = {k: None for k in self.code_modalities}
                ct_e = {k: None for k in self.code_modalities}
                ct_d = {k: None for k in self.code_modalities}
                st_hat = {k: None for k in self.code_modalities}
                p_attns = {k: None for k in self.code_modalities}

                for i, modal in enumerate(self.code_modalities):
                    # output_attn: batch_size*rnn_hidden_size
                    # ct_e: batch_size*rnn_hidden_size
                    # ct_d: batch_size*rnn_hidden_size
                    # st_hat: batch_size*rnn_hidden_size
                    # p_attn: batch_size*code_len
                    output_attns[modal], sum_temporal_srcs[modal], prev_s[modal], ct_e[modal], ct_d[modal], st_hat[
                        modal], p_attns[modal] = \
                        self.attns[i](input_emb, enc_output[modal], enc_padding_mask[modal],
                                      sum_temporal_srcs[modal], dec_hidden, prev_s[modal])
                # fuse multi-modality
                if len(self.code_modalities) > 1:
                    output_attns_final = torch.tanh(self.fuse_linear(
                        torch.cat([output_attns[key] for key in sorted(output_attns.keys())],
                                  dim=1)))  # batch_size*rnn_hidden_size
                else:  # single modality
                    output_attns_final = output_attns[self.code_modalities[0]]
                # concatenate context vector and hidden state
                final_out = torch.tanh(self.concat_map(
                    torch.cat((output_attns_final, dec_hidden[0][-1]), dim=1)))  # batch_size*rnn_hidden_size
                final_out = F.dropout(final_out, self.dropout, training=self.training)
                decoded = self.generate_linear(final_out)  # (batch_size*comment_dict_size)
                logprobs = F.log_softmax(decoded, dim=1)  # (batch_size*comment_dict_size)
                prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)

                if self.pointer:
                    p_gen = torch.cat([ct_e['tok'], ct_d['tok'], st_hat['tok'], input_emb.squeeze(1)], 1)
                    p_gen = self.p_gen_linear(p_gen)
                    p_gen = torch.sigmoid(p_gen)
                    prob_prev = p_gen * prob_prev
                    p_attn_ = (1 - p_gen) * p_attns['tok']
                    if pointer_extra_zeros is not None:
                        prob_prev = torch.cat([prob_prev, pointer_extra_zeros], dim=1)
                    prob_prev = prob_prev.scatter_add(1, code_dict_comment, p_attn_)
                    logprobs = torch.log(prob_prev + EPS_ZERO)  # self.opt.eps,  + 1e-12
            else:
                raise NotImplementedError

            if sample_max:
                sample_logprobs, predicted = torch.max(logprobs, 1)
                seq[:, t] = predicted.reshape(-1)
                seq_logp_gathered[:, t] = sample_logprobs
                seq_logprobs[:, t, :] = logprobs
            else:
                predicted = torch.multinomial(prob_prev, 1)  # .to(device)
                seq[:, t] = predicted.reshape(-1)
                seq_logp_gathered[:, t] = logprobs.gather(1, predicted).reshape(-1)
                seq_logprobs[:, t, :] = logprobs
            # seq_lprob_sum[:, t] = logprobs.sum(dim=-1, keepdim=True)
            seq_lprob_sum[:, t] = logprobs.sum(dim=-1)
            input = predicted.reshape(-1, 1)
            is_oov = (input >= self.token_num).long().to(device)
            LOGGER.debug('is_oov: {}'.format(is_oov.size()))
            input = (1 - is_oov) * input.detach() + (is_oov) * UNK

        return seq, seq_logprobs, seq_logp_gathered, seq_lprob_sum  # , comment_target_padded,

    # TODO: There are still some bugs in the beam search.
    def sample_beam(self, input, hidden_state, sample_opt={}):
        # print('sample_beam')
        beam_size = sample_opt.get('beam_size', 10)
        batch_size = input.size(1)
        seq_length = sample_opt.get('seq_length', self.max_predict_length)
        # print('beam_size: ', beam_size)
        # print('batch_size: ', batch_size)
        # assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # seq_all = torch.LongTensor(self.seq_length, batch_size, beam_size).zero_()
        seq_all = torch.LongTensor(batch_size, seq_length, beam_size).zero_()
        # seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seq = torch.LongTensor(batch_size, seq_length).zero_()
        # seqLogprobs = torch.FloatTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, seq_length).zero_()
        # print('seq_all: ', seq_all.type(), seq_all.size())
        # print(seq_all)
        # print('seq: ', seq.type(), seq.size())
        # print(seq)
        # print('seqLogprobs: ', seqLogprobs.type(), seqLogprobs.size())
        # print(seqLogprobs)

        # lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        # print('self.done_beams: ', self.done_beams)
        for k in range(batch_size):
            # copy the hidden state for beam_size time.
            state = []
            for state_tmp in hidden_state:
                state.append(state_tmp[:, k, :].reshape(1, 1, -1).expand(1, beam_size, self.hidden_size).clone())
            # print('state: ')
            # print(state)
            state = tuple(state)
            beam_seq = torch.LongTensor(seq_length, beam_size).zero_()
            # print('beam_seq: ', beam_seq.type(), beam_seq.size())
            # print(beam_seq)
            beam_seq_logprobs = torch.FloatTensor(seq_length, beam_size).zero_()
            # print('beam_seq_logprobs: ', beam_seq_logprobs.type(), beam_seq_logprobs.size())
            # print(beam_seq_logprobs)
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
            # print('beam_logprobs_sum: ', beam_logprobs_sum.type(), beam_logprobs_sum.size())
            for t in range(seq_length + 1):
                # print('step-t: ', t)
                if t == 0:  # input <bos>
                    it = input.resize_(1, beam_size).fill_(BOS)  # .data
                    xt = self.wemb(it.detach())
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float().cpu()  # lets go to CPU for more efficiency in indexing operations
                    ys, ix = torch.sort(logprobsf, 1,
                                        True)  # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:  # at first time step only the first beam is active
                        rows = 1
                    for cc in range(cols):  # for each column (word, essentially)
                        for qq in range(rows):  # for each beam expansion
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[qq, cc]
                            if beam_seq[t - 2, qq] == self.embed_size:  # self.opt.ninp:
                                local_logprob.data.fill_(-9999)
                            # print('local_logprob: ', local_logprob.type(), local_logprob.size())
                            # print(local_logprob)
                            # print('beam_logprobs_sum[qq]: ', beam_logprobs_sum[qq].type(), beam_logprobs_sum[qq].size())
                            # print(beam_logprobs_sum[qq])
                            candidate_logprob = beam_logprobs_sum[qq] + local_logprob
                            candidates.append({'c': ix.data[qq, cc], 'q': qq, 'p': candidate_logprob.item(),
                                               'r': local_logprob.item()})

                    candidates = sorted(candidates, key=lambda x: -x['p'])
                    # print('candidates: ', candidates)
                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    # print('new_state: ')
                    # print(new_state)
                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == self.opt.ninp or t == seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix]
                                                       })

                    # encode as vectors
                    it = beam_seq[t - 1].reshape(1, -1)
                    xt = self.wemb(it.cuda())

                if t >= 1:
                    state = new_state

                output, state = self.rnn(xt, hidden=state)

                output = F.dropout(output, self.dropout, training=self.training)
                decoded = self.generate_linear(output.reshape(output.size(0) * output.size(1), output.size(2)))
                logprobs = F.log_softmax(decoded, dim=1)  # self.beta *

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            for ii in range(beam_size):
                seq_all[:, k, ii] = self.done_beams[k][ii]['seq']

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def step_bak(self, input, enc_output, dec_hidden, enc_padding_mask, code_dict_comment=None,
                 pointer_extra_zeros=None, prev_s=None):
        # For attention
        if self.attn_type:  # only attention in ['general', 'dot', 'mlp', 'intra'], no pointer initialization
            if isinstance(self.code_modalities, str):  # for uni modality
                enc_output = enc_output[self.code_modalities]  # (batch_size*m_len*rnn_hidden_size)
                enc_padding_mask = enc_padding_mask[self.code_modalities]  # (batch_size*m_len): 0 or 1
            elif isinstance(self.code_modalities, list):  # for multi modalities
                pass
        if self.attn_type == 'intra':  # attention and pointer initialization
            sum_temporal_srcs = None
        elif self.attn_type is None:
            LOGGER.debug('no attention in {}'.format(self.__class__.__name__))
            pass

        # For pointer
        # if self.pointer:
        #     code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
        # print('self.token_num: ', self.token_num)
        # print('pointer_extra_zeros.size(1): ', pointer_extra_zeros.size(1))
        # seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num + pointer_extra_zeros.size(1)).to(
        #     device)
        # else:
        #     # seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)
        #     pass

        if self.attn_type is None:
            input_emb = self.wemb(input)  # (batch_size*1*emb_size)
            output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
            decoded = self.linear(output.reshape(-1, output.size(2)))  # (batch_size*comment_dict_size)
            logprobs = F.log_softmax(decoded, dim=-1)  # (batch_size*comment_dict_size)
            prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)
        elif self.attn_type in ['general', 'dot', 'mlp']:
            input_emb = self.wemb(input)  # (batch_size*1*emb_size)
            output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
            output = output.squeeze(1)  # one step decode (batch_size*rnn_hidden_size, )
            output_attn, p_attn = self.attn(output, enc_output,
                                            enc_padding_mask)  # (batch_size*rnn_hidden_size, batch_size*code_len)
            decoded = self.linear(output_attn)  # (batch_size*comment_dict_size)
            logprobs = F.log_softmax(decoded, dim=1)  # (batch_size*comment_dict_size)
            prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)
            if self.pointer:
                st_hat = dec_hidden[0][-1]  # (batch_size*rnn_hidden_size)
                p_gen_input = torch.cat([output_attn, st_hat, input_emb.squeeze(1)],
                                        1)  # B x (2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = torch.sigmoid(p_gen)
                prob_prev_ = p_gen * prob_prev
                p_attn_ = (1 - p_gen) * p_attn
                if pointer_extra_zeros is not None:
                    prob_prev_ = torch.cat([prob_prev_, pointer_extra_zeros], 1)

                prob_prev = prob_prev_.scatter_add(1, code_dict_comment, p_attn_)
                logprobs = torch.log(prob_prev + EPS_ZERO)  # self.opt.eps,  + 1e-12

        elif self.attn_type == 'intra':
            input_emb = self.wemb(input)  # (batch_size*1*emb_size)
            output, dec_hidden = self.rnn(input_emb, hidden=dec_hidden)  # (batch_size*1*rnn_hidden_size, )
            # output_attn: batch_size*rnn_hidden_size
            # ct_e: batch_size*rnn_hidden_size
            # ct_d: batch_size*rnn_hidden_size
            # st_hat: batch_size*rnn_hidden_size
            # p_attn: batch_size*code_len
            output_attn, sum_temporal_srcs, prev_s, ct_e, ct_d, st_hat, p_attn = \
                self.attn(input_emb, enc_output, enc_padding_mask,
                          sum_temporal_srcs, dec_hidden, prev_s)
            decoded = self.linear(output_attn)  # (batch_size*comment_dict_size)
            logprobs = F.log_softmax(decoded, dim=1)  # (batch_size*comment_dict_size)
            prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)

            if self.pointer:
                p_gen = torch.cat([ct_e, ct_d, st_hat, input_emb.squeeze(1)], 1)
                p_gen = self.p_gen_linear(p_gen)
                p_gen = torch.sigmoid(p_gen)
                prob_prev = p_gen * prob_prev
                p_attn_ = (1 - p_gen) * p_attn
                if pointer_extra_zeros is not None:
                    # print('pointer_extra_zeros-: ', pointer_extra_zeros.size())
                    # print(pointer_extra_zeros)
                    prob_prev = torch.cat([prob_prev, pointer_extra_zeros], dim=1)
                # print('p_attn_: ', p_attn_.size())
                # print(p_attn_)
                prob_prev = prob_prev.scatter_add(1, code_dict_comment, p_attn_)
                logprobs = torch.log(prob_prev + EPS_ZERO)  # self.opt.eps,  + 1e-12
        else:
            raise NotImplementedError

        return logprobs, prob_prev, dec_hidden

    def forward_bak(self, batch, enc_output, dec_hidden, enc_padding_mask, sample_opt={}) -> Tuple:
        sample_max, seq_length = sample_opt.get('sample_max', 1), \
                                 sample_opt.get('seq_length', self.max_predict_length)
        # print('batch...')
        # pprint(batch)
        # (batch_size*mLen, batch_size*(mLen+1),)
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        device = comment.device
        batch_size = comment.size(0)
        seq_length = min(comment.size(1) + 1, seq_length)  # +1 is for EOS
        input = torch.zeros(batch_size, 1).long().fill_(BOS).to(device)  # (batch_size*1) and all items are 2 (BOS)
        # Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise
        mask = torch.LongTensor(batch_size).fill_(1).to(device)  # (batch_size) and all items are 1
        seq, seq_logp_gathered, seq_lprob_sum = torch.zeros(batch_size, seq_length).long().to(device), \
                                                torch.zeros(batch_size, seq_length).to(device), \
                                                torch.zeros(batch_size, seq_length).to(device)

        # For pointer
        if self.pointer:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
            # print('self.token_num: ', self.token_num)
            # print('pointer_extra_zeros.size(1): ', pointer_extra_zeros.size(1))
            seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num + pointer_extra_zeros.size(1)).to(
                device)
        else:
            seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)
            code_dict_comment, pointer_extra_zeros = None, None

        prev_s, seq_padding_mask = None, []
        for t in range(seq_length):
            # TODO: torch.rand(batch_size) >= 0 is all True. Done: >=0 is used for debug.
            # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            use_ground_truth = (torch.rand(batch_size) > 0.25).unsqueeze(1).long().to(device)
            # Select decoder input based on use_ground_truth probabilities
            input = use_ground_truth * comment_input[:, t].unsqueeze(1) + (1 - use_ground_truth) * input

            logprobs, prob_prev, dec_hidden = self.step(input, enc_output, dec_hidden, enc_padding_mask,
                                                        code_dict_comment, pointer_extra_zeros)
            if sample_max:
                sample_logprobs, predicted = torch.max(logprobs, 1)
                seq[:, t] = predicted.reshape(-1)
                seq_logp_gathered[:, t] = sample_logprobs
                seq_logprobs[:, t, :] = logprobs
            else:
                predicted = torch.multinomial(prob_prev, 1)  # .to(device)
                seq[:, t] = predicted.reshape(-1)
                seq_logp_gathered[:, t] = logprobs.gather(1, predicted).reshape(-1)
                seq_logprobs[:, t, :] = logprobs

            seq_lprob_sum[:, t] = logprobs.sum(dim=-1)
            input = predicted.reshape(-1, 1)
            # .detach() Mask indicating whether sampled word is OOV
            is_oov = (input >= self.token_num).long().to(device).detach()
            input = (1 - is_oov) * input + (is_oov) * UNK

            mask_t = torch.zeros(batch_size).to(device)  # Padding mask of batch for current time step
            mask_t[
                mask == 1] = 1  # If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            # If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            mask[(mask == 1).int() + (input.squeeze(1) == EOS).int() == 2] = 0
            seq_padding_mask.append(mask_t)

        seq_padding_mask = torch.stack(seq_padding_mask).t()  # (batch_size*max_len)

        return seq, seq_logprobs, seq_logp_gathered, seq_padding_mask, seq_lprob_sum, dec_hidden,
