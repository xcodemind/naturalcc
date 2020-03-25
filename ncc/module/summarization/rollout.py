# -*- coding: utf-8 -*-

import sys

sys.path.append('..')

import numpy as np
import torch
import torch.nn.functional as F
from ncc.utils.constants import *
import ncc.utils.util
import copy


class Rollout:
    def __init__(self):  # , max_sentence_length, corpus
        self.rnn = None
        self.linear = None

    def reward_pg(self, wemb, code_batch, code_length, comment_batch, dict_comment, dec_hidden, generated,
                  current_padding_mask, seq_length, rollout_num, steps=1):
        assert rollout_num % steps == 0, "Monte Carlo Count can't be divided by Steps"
        rollout_num //= steps
        print('reward_pg..')
        print('rollout-rollout_num: ', rollout_num)
        print('rollout-code_batch: ', code_batch.type(), code_batch.size())
        print(code_batch)
        print('rollout-code_length: ')
        print(code_length)
        print('rollout-comment_batch: ', comment_batch.type(), comment_batch.size())
        print(comment_batch)
        print('rollout-generated: ', generated.type(), generated.size())
        print(generated)
        print('rollout-current_padding_mask: ', len(current_padding_mask))
        print(current_padding_mask[0])
        current_padding_mask = torch.stack(current_padding_mask)
        print('rollout-current_padding_mask: ', current_padding_mask.type(), current_padding_mask.size())
        print(current_padding_mask)
        with torch.no_grad():
            print('rollout-generated: ', generated.type(), generated.size())
            print(generated)
            batch_size = generated.size(1)
            print('rollout-batch_size: ', batch_size)
            # result = torch.zeros(batch_size, 1).cuda()
            # print('rollout-result: ', result.type(), result.size())
            # print(result)
            remaining = seq_length - generated.shape[0]
            print('rollout-remaining: ', remaining)
            print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
            h, c = dec_hidden
            generated = generated.repeat(1, rollout_num)
            print('rollout-generated-: ', generated.type(), generated.size())
            print(generated)
            # print('rollout-steps: ', steps)

            # for _ in range(steps):
            dec_hidden = (h.repeat(1, rollout_num, 1), c.repeat(1, rollout_num, 1))
            # print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
            # print(dec_hidden)
            input = generated[-1, :].unsqueeze(1).t()
            # print('rollout-input: ', input.type(), input.size())
            # print(input)
            current_generated = generated
            rollout_padding_mask = []
            mask = torch.LongTensor(rollout_num * batch_size).fill_(1).cuda()
            print('rollout-mask: ', mask.type(), mask.size())
            print(mask)
            for i in range(remaining):
                input_emb = wemb(input)
                # print('rollout-input_emb: ', input_emb.type(), input_emb.size())
                # print(input_emb)
                output, dec_hidden = self.rnn(input_emb, dec_hidden)
                # print('rollout-output: ', output.type(), output.size())
                # print(output)
                # output = F.dropout(output, self.d, training=self.training)
                # print('output-: ', output.type(), output.size())
                # print(output)
                decoded = self.linear(output.reshape(-1, output.size(2)))
                # print('rollout-decoded: ', decoded.type(), decoded.size())
                # print(decoded)
                logprobs = F.log_softmax(decoded, dim=1)  # self.beta *
                prob_prev = torch.exp(logprobs)
                # print('rollout-prob_prev: ', prob_prev.type(), prob_prev.size())
                # print(prob_prev)
                predicted = torch.multinomial(prob_prev, 1)
                # sample_logprobs, predicted = torch.max(logprobs, 1)
                # print('rollout-predicted: ', predicted.type(), predicted.size())
                # print(predicted)
                # embed the next input, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                input = predicted.reshape(1, -1)
                # print('rollout-input-: ', input.type(), input.size())
                # print(input)
                # print('rollout-current_generated: ', current_generated.type(), current_generated.size())
                # print(current_generated)
                mask_t = torch.zeros(rollout_num * batch_size).cuda()  # Padding mask of batch for current time step
                # print('forward_pg-mask_t: ', mask_t.type(), mask_t.size())
                # print(mask_t)
                mask_t[
                    mask == 1] = 1  # If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
                print('forward_pg-mask_t-: ', mask_t.type(), mask_t.size())
                print(mask_t)
                print('forward_pg-mask: ', mask.type(), mask.size())
                print(mask)
                print('forward_pg-(mask == 1): ')
                print((mask == 1))
                print('forward_pg-(input == constants.EOS) == 2: ')
                print((input == constants.EOS) == 2)
                mask[(mask == 1) + (
                        input.squeeze() == constants.EOS) == 2] = 0  # If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
                print('forward_pg-mask-: ', mask.type(), mask.size())
                print(mask)
                rollout_padding_mask.append(mask_t)

                current_generated = torch.cat([current_generated, input], dim=0)
                print('rollout-current_generated-: ', current_generated.type(), current_generated.size())
                print(current_generated)
            print('rollout-current_generated-final: ', current_generated.type(), current_generated.size())
            print(current_generated)
            # current_generated_cleaned = self.clean_generated_comment(current_generated)
            # print('rollout-current_generated_cleaned-final: ', current_generated_cleaned.type(), current_generated_cleaned.size())
            # print(current_generated_cleaned)
            # print('rollout-current_generated_cleaned-final[0:10]: ', current_generated_cleaned[0:10].type(), current_generated_cleaned[0:10].size())
            # print(current_generated_cleaned[0:10])

            current_padding_mask = current_padding_mask.repeat(1, rollout_num)
            print('rollout-current_padding_mask-: ', current_padding_mask.type(), current_padding_mask.size())
            print(current_padding_mask)

            if rollout_padding_mask:
                rollout_padding_mask = torch.stack(rollout_padding_mask)
                print('rollout-rollout_padding_mask-: ', rollout_padding_mask.type(), rollout_padding_mask.size())
                print(rollout_padding_mask)
            current_padding_mask = torch.cat([current_padding_mask, rollout_padding_mask]).byte()
            print('rollout-current_padding_mask--: ', current_padding_mask.type(), current_padding_mask.size())
            print(current_padding_mask)
            # assert False
            comment_fake = torch.masked_select(current_generated, current_padding_mask)
            print('rollout-comment_fake: ', comment_fake.type(), comment_fake.size())
            print(comment_fake)
            reward = disc(comment_fake)
            print('rollout-reward: ', rollout.type(), rollout.size())
            print(reward)

            reward = reward.reshape(-1, rollout_num, batch_size)
            print('rollout-reward-: ', rollout.type(), rollout.size())
            print(reward)
            reward = reward.sum(1)
            print('rollout-reward--: ', rollout.type(), rollout.size())
            print(reward)
            reward = reward / 5
            print('rollout-reward---: ', rollout.type(), rollout.size())
            print(reward)


            rewards = []
            for r in range(rollout_num):
                # comment_fake = current_generated_cleaned.reshape(batch_size, rollout_num, -1)[:,r,:].t()
                comment_fake = current_generated.reshape(-1, rollout_num, batch_size)[:, r, :]
                # print('rollout-r: ', r)
                # print('rollout-comment_fake: ', comment_fake.type(), comment_fake.size())
                # print(comment_fake)
                # reward = disc(comment_fake)  # bleu score code_batch, code_length,
                # reward, pred = sent_reward_func(comment_fake.t().cpu().numpy().tolist(), comment_batch.t().cpu().numpy().tolist())
                comment_fake_list, comment_batch_list = comment_fake.t().tolist(), comment_batch[1:].t().tolist()
                # print('forward_pg-seq_list: ', len(seq_list), utils.util.clean_up_sentence(seq_list[0], remove_unk=False, remove_eos=True))
                # print('forward_pg-comment_batch_list: ', len(comment_batch_list), utils.util.clean_up_sentence(comment_batch_list[0], remove_unk=False, remove_eos=True))
                reward = []
                for idx in range(len(comment_fake_list)):
                    res = {0: [' '.join(dict_comment.getLabel(i) for i in
                                        utils.util.clean_up_sentence(comment_fake_list[idx], remove_unk=False,
                                                                     remove_eos=True))]}
                    gts = {0: [' '.join(dict_comment.getLabel(i) for i in
                                        utils.util.clean_up_sentence(comment_batch_list[idx], remove_unk=False,
                                                                     remove_eos=True))]}
                    # res, gts = {k: [' '.join(str(i) for i in seq_list[k])] for k in range(len(seq_list))}, {k: [' '.join(str(i) for i in comment_batch_list[k])] for k in range(len(comment_batch_list))}
                    # if idx < 4:
                    #     print('forward_pg-res: ', res[0])
                    #     print('forward_pg-gts: ', gts[0])
                    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
                    # print('Bleu-1: ', np.mean(scores_Bleu[0]))
                    # print('forward_pg-scores_Bleu: ', scores_Bleu)
                    reward.append(np.mean(scores_Bleu[0]))
                # reward, pred = sent_reward_func(seq.t().tolist(), comment_batch.t().tolist())
                # print('forward_pg-reward: ')
                # print(reward)
                reward = torch.Tensor(reward)  # .repeat(seq_length, 1).cuda()
                # print('rollout-reward: ')
                # print(reward)
                rewards.append(reward)
                # reward = reward.reshape(batch_size, rollout_num, -1).sum(1)
                # print('rollout-reward--: ', reward.type(), reward.size())
                # print(reward)
            rewards = torch.stack(rewards, 0)
            # print('rollout-rewards-: ', rewards.type(), rewards.size())
            # print(rewards)
            # rewards = rewards.reshape(batch_size, rollout_num, -1).sum(1)
            # rewards = rewards.reshape(rollout_num, batch_size, -1).sum(0)
            rewards = rewards.sum(0).reshape(-1, 1)
            # print('rollout-rewards--: ', rewards.type(), rewards.size())
            # print(rewards)
            # print('rollout-result-: ', result.type(), result.size())
            # print(result)
            # result += rewards
            # result /= rollout_num
            # print('rollout-result--: ', result.type(), result.size())
            # print(result)
            rewards = rewards / rollout_num

            return rewards

    def reward_a2c(self, wemb, disc, code_batch, code_length, dec_hidden, generated, seq_length, rollout_num, steps=1):
        assert rollout_num % steps == 0, "Monte Carlo Count can't be divided by Steps"
        rollout_num //= steps
        print('rollout-rollout_num: ', rollout_num)
        print('rollout-code_batch: ', code_batch.type(), code_batch.size())
        print(code_batch)
        print('rollout-code_length: ')
        print(code_length)
        with torch.no_grad():
            print('rollout-generated: ', generated.type(), generated.size())
            print(generated)
            batch_size = generated.size(0)
            # print('rollout-batch_size: ', batch_size)
            result = torch.zeros(batch_size, 1).cuda()
            print('rollout-result: ', result.type(), result.size())
            print(result)
            remaining = seq_length - generated.shape[1]
            print('rollout-remaining: ', remaining)
            print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
            h, c = dec_hidden
            generated = generated.repeat(rollout_num, 1)
            print('rollout-generated-: ', generated.type(), generated.size())
            print(generated)
            print('rollout-steps: ', steps)
            for _ in range(steps):
                dec_hidden = (h.repeat(1, rollout_num, 1), c.repeat(1, rollout_num, 1))
                print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
                print(dec_hidden)
                input = generated[:, -1].unsqueeze(1).t()
                print('rollout-input: ', input.type(), input.size())
                print(input)
                current_generated = generated
                for i in range(remaining):
                    input_emb = wemb(input)
                    # print('rollout-input_emb: ', input_emb.type(), input_emb.size())
                    # print(input_emb)
                    output, dec_hidden = self.rnn(input_emb, dec_hidden)
                    print('rollout-output: ', output.type(), output.size())
                    print(output)
                    # output = F.dropout(output, self.d, training=self.training)
                    # print('output-: ', output.type(), output.size())
                    # print(output)
                    decoded = self.linear(output.reshape(-1, output.size(2)))
                    print('rollout-decoded: ', decoded.type(), decoded.size())
                    print(decoded)
                    logprobs = F.log_softmax(decoded, dim=1)  # self.beta *
                    prob_prev = torch.exp(logprobs)
                    print('rollout-prob_prev: ', prob_prev.type(), prob_prev.size())
                    print(prob_prev)
                    predicted = torch.multinomial(prob_prev, 1)
                    # sample_logprobs, predicted = torch.max(logprobs, 1)
                    print('rollout-predicted: ', predicted.type(), predicted.size())
                    print(predicted)
                    # embed the next input, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                    input = predicted.reshape(1, -1)
                    print('rollout-input-: ', input.type(), input.size())
                    print(input)
                    print('rollout-current_generated: ', current_generated.type(), current_generated.size())
                    print(current_generated)
                    current_generated = torch.cat([current_generated, predicted.reshape(-1, 1)], dim=1)
                    print('rollout-current_generated-: ', current_generated.type(), current_generated.size())
                    print(current_generated)

                print('rollout-current_generated-final: ', current_generated.type(), current_generated.size())
                print(current_generated)

                current_generated_cleaned = self.clean_generated_comment(current_generated)
                print('rollout-current_generated_cleaned-final: ', current_generated_cleaned.type(),
                      current_generated_cleaned.size())
                print(current_generated_cleaned)
                print('rollout-current_generated_cleaned-final[0:10]: ', current_generated_cleaned[0:10].type(),
                      current_generated_cleaned[0:10].size())
                print(current_generated_cleaned[0:10])

                rewards = []
                for r in range(rollout_num):
                    # comment_fake = current_generated_cleaned.reshape(batch_size, rollout_num, -1)[:,r,:].t()
                    comment_fake = current_generated_cleaned.reshape(rollout_num, batch_size, -1)[r, :, :].t()
                    reward = disc(comment_fake)  # bleu score code_batch, code_length,
                    print('rollout-reward: ', reward.type(), reward.size())
                    print(reward)
                    reward = torch.exp(reward[:, 1])
                    # reward = reward[:, 1]
                    print('rollout-reward-: ', reward.type(), reward.size())
                    print(reward)
                    rewards.append(reward)
                    # reward = reward.reshape(batch_size, rollout_num, -1).sum(1)
                    print('rollout-reward--: ', reward.type(), reward.size())
                    print(reward)
                rewards = torch.stack(rewards, 0)
                print('rollout-rewards-: ', rewards.type(), rewards.size())
                print(rewards)
                # rewards = rewards.reshape(batch_size, rollout_num, -1).sum(1)
                # rewards = rewards.reshape(rollout_num, batch_size, -1).sum(0)
                rewards = rewards.sum(0).reshape(-1, 1)
                print('rollout-rewards--: ', rewards.type(), rewards.size())
                print(rewards)
                # print('rollout-result-: ', result.type(), result.size())
                # print(result)
                result += rewards
                result /= rollout_num
                print('rollout-result--: ', result.type(), result.size())
                print(result)
            return result

    def reward_gan_bak(self, wemb, disc, code_batch, code_length, dec_hidden, generated, seq_length, rollout_num,
                       steps=1):
        assert rollout_num % steps == 0, "Monte Carlo Count can't be divided by Steps"
        rollout_num //= steps
        # print('rollout-rollout_num: ', rollout_num)
        # print('rollout-code_batch: ', code_batch.type(), code_batch.size())
        # print(code_batch)
        # print('rollout-code_length: ')
        # print(code_length)
        with torch.no_grad():
            # print('rollout-generated: ', generated.type(), generated.size())
            # print(generated)
            batch_size = generated.size(0)
            # print('rollout-batch_size: ', batch_size)
            result = torch.zeros(batch_size, 1).cuda()
            # print('rollout-result: ', result.type(), result.size())
            # print(result)
            remaining = seq_length - generated.shape[1]
            # print('rollout-remaining: ', remaining)
            # print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
            h, c = dec_hidden
            generated = generated.repeat(rollout_num, 1)
            # print('rollout-generated-: ', generated.type(), generated.size())
            # print(generated)
            # print('rollout-steps: ', steps)
            for _ in range(steps):
                dec_hidden = (h.repeat(1, rollout_num, 1), c.repeat(1, rollout_num, 1))
                # print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
                # print(dec_hidden)
                input = generated[:, -1].unsqueeze(1).t()
                # print('rollout-input: ', input.type(), input.size())
                # print(input)
                current_generated = generated
                for i in range(remaining):
                    input_emb = wemb(input)
                    # print('rollout-input_emb: ', input_emb.type(), input_emb.size())
                    # print(input_emb)
                    output, dec_hidden = self.rnn(input_emb, dec_hidden)
                    # print('rollout-output: ', output.type(), output.size())
                    # print(output)
                    # output = F.dropout(output, self.d, training=self.training)
                    # print('output-: ', output.type(), output.size())
                    # print(output)
                    decoded = self.linear(output.reshape(-1, output.size(2)))
                    # print('rollout-decoded: ', decoded.type(), decoded.size())
                    # print(decoded)
                    logprobs = F.log_softmax(decoded, dim=1)  # self.beta *
                    prob_prev = torch.exp(logprobs)
                    # print('rollout-prob_prev: ', prob_prev.type(), prob_prev.size())
                    # print(prob_prev)
                    predicted = torch.multinomial(prob_prev, 1)
                    # sample_logprobs, predicted = torch.max(logprobs, 1)
                    # print('rollout-predicted: ', predicted.type(), predicted.size())
                    # print(predicted)
                    # embed the next input, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                    input = predicted.reshape(1, -1)
                    # print('rollout-input-: ', input.type(), input.size())
                    # print(input)
                    # print('rollout-current_generated: ', current_generated.type(), current_generated.size())
                    # print(current_generated)
                    current_generated = torch.cat([current_generated, predicted.reshape(-1, 1)], dim=1)
                    # print('rollout-current_generated-: ', current_generated.type(), current_generated.size())
                    # print(current_generated)

                # print('rollout-current_generated-final: ', current_generated.type(), current_generated.size())
                # print(current_generated)

                current_generated_cleaned = self.clean_generated_comment(current_generated)
                # print('rollout-current_generated_cleaned-final: ', current_generated_cleaned.type(), current_generated_cleaned.size())
                # print(current_generated_cleaned)
                # print('rollout-current_generated_cleaned-final[0:10]: ', current_generated_cleaned[0:10].type(), current_generated_cleaned[0:10].size())
                # print(current_generated_cleaned[0:10])

                rewards = []
                for r in range(rollout_num):
                    # comment_fake = current_generated_cleaned.reshape(batch_size, rollout_num, -1)[:,r,:].t()
                    comment_fake = current_generated_cleaned.reshape(rollout_num, batch_size, -1)[r, :, :].t()
                    reward = disc(comment_fake)  # bleu score code_batch, code_length,
                    # print('rollout-reward: ', reward.type(), reward.size())
                    # print(reward)
                    reward = torch.exp(reward[:, 1])
                    # reward = reward[:, 1]
                    # print('rollout-reward-: ', reward.type(), reward.size())
                    # print(reward)
                    rewards.append(reward)
                    # reward = reward.reshape(batch_size, rollout_num, -1).sum(1)
                    # print('rollout-reward--: ', reward.type(), reward.size())
                    # print(reward)
                rewards = torch.stack(rewards, 0)
                # print('rollout-rewards-: ', rewards.type(), rewards.size())
                # print(rewards)
                # rewards = rewards.reshape(batch_size, rollout_num, -1).sum(1)
                # rewards = rewards.reshape(rollout_num, batch_size, -1).sum(0)
                rewards = rewards.sum(0).reshape(-1, 1)
                # print('rollout-rewards--: ', rewards.type(), rewards.size())
                # print(rewards)
                # print('rollout-result-: ', result.type(), result.size())
                # print(result)
                result += rewards
                result /= rollout_num
                # print('rollout-result--: ', result.type(), result.size())
                # print(result)
            return result

    def reward_gan(self, wemb, disc, code_batch, code_length, comment_batch, dict_comment, dec_hidden, generated,
                   current_padding_mask, seq_length, rollout_num, steps=1):
        assert rollout_num % steps == 0, "Monte Carlo Count can't be divided by Steps"
        rollout_num //= steps
        # print('reward_pg..')
        # print('rollout-rollout_num: ', rollout_num)
        # print('rollout-code_batch: ', code_batch.type(), code_batch.size())
        # print(code_batch)
        # print('rollout-code_length: ')
        # print(code_length)
        # print('rollout-comment_batch: ', comment_batch.type(), comment_batch.size())
        # print(comment_batch)
        # print('rollout-generated: ', generated.type(), generated.size())
        # print(generated)
        # print('rollout-current_padding_mask: ', len(current_padding_mask))
        # print(current_padding_mask[0])
        current_padding_mask = torch.stack(current_padding_mask)
        # print('rollout-current_padding_mask: ', current_padding_mask.type(), current_padding_mask.size())
        # print(current_padding_mask)
        with torch.no_grad():
            # print('rollout-generated: ', generated.type(), generated.size())
            # print(generated)
            batch_size = generated.size(1)
            # print('rollout-batch_size: ', batch_size)
            # result = torch.zeros(batch_size, 1).cuda()
            # print('rollout-result: ', result.type(), result.size())
            # print(result)
            remaining = seq_length - generated.shape[0]
            # print('rollout-remaining: ', remaining)
            # print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
            h, c = dec_hidden
            generated = generated.repeat(1, rollout_num)
            # print('rollout-generated-: ', generated.type(), generated.size())
            # print(generated)
            # print('rollout-steps: ', steps)

            # for _ in range(steps):
            dec_hidden = (h.repeat(1, rollout_num, 1), c.repeat(1, rollout_num, 1))
            # print('rollout-dec_hidden: ', dec_hidden[0].type(), dec_hidden[0].size())
            # print(dec_hidden)
            input = generated[-1, :].unsqueeze(1).t()
            # print('rollout-input: ', input.type(), input.size())
            # print(input)
            current_generated = generated
            rollout_padding_mask = []
            mask = torch.LongTensor(rollout_num * batch_size).fill_(1).cuda()
            # print('rollout-mask: ', mask.type(), mask.size())
            # print(mask)
            for i in range(remaining):
                input_emb = wemb(input)
                # print('rollout-input_emb: ', input_emb.type(), input_emb.size())
                # print(input_emb)
                output, dec_hidden = self.rnn(input_emb, dec_hidden)
                # print('rollout-output: ', output.type(), output.size())
                # print(output)
                # output = F.dropout(output, self.d, training=self.training)
                # print('output-: ', output.type(), output.size())
                # print(output)
                decoded = self.linear(output.reshape(-1, output.size(2)))
                # print('rollout-decoded: ', decoded.type(), decoded.size())
                # print(decoded)
                logprobs = F.log_softmax(decoded, dim=1)  # self.beta *
                prob_prev = torch.exp(logprobs)
                # print('rollout-prob_prev: ', prob_prev.type(), prob_prev.size())
                # print(prob_prev)
                predicted = torch.multinomial(prob_prev, 1)
                # sample_logprobs, predicted = torch.max(logprobs, 1)
                # print('rollout-predicted: ', predicted.type(), predicted.size())
                # print(predicted)
                # embed the next input, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                input = predicted.reshape(1, -1)
                # print('rollout-input-: ', input.type(), input.size())
                # print(input)
                # print('rollout-current_generated: ', current_generated.type(), current_generated.size())
                # print(current_generated)
                mask_t = torch.zeros(rollout_num * batch_size).cuda()  # Padding mask of batch for current time step
                # print('forward_pg-mask_t: ', mask_t.type(), mask_t.size())
                # print(mask_t)
                mask_t[
                    mask == 1] = 1  # If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
                # print('forward_pg-mask_t-: ', mask_t.type(), mask_t.size())
                # print(mask_t)
                # print('forward_pg-mask: ', mask.type(), mask.size())
                # print(mask)
                # print('forward_pg-(mask == 1): ')
                # print((mask == 1))
                # print('forward_pg-(input == data.codesum.constants.EOS) == 2: ')
                # print((input == data.codesum.constants.EOS) == 2)
                mask[(mask == 1) + (
                        input.squeeze() == constants.EOS) == 2] = 0  # If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
                # print('forward_pg-mask-: ', mask.type(), mask.size())
                # print(mask)
                rollout_padding_mask.append(mask_t)

                current_generated = torch.cat([current_generated, input], dim=0)
                # print('rollout-current_generated-: ', current_generated.type(), current_generated.size())
                # print(current_generated)
            # print('rollout-current_generated-final: ', current_generated.type(), current_generated.size())
            # print(current_generated)
            # current_generated_cleaned = self.clean_generated_comment(current_generated)
            # print('rollout-current_generated_cleaned-final: ', current_generated_cleaned.type(), current_generated_cleaned.size())
            # print(current_generated_cleaned)
            # print('rollout-current_generated_cleaned-final[0:10]: ', current_generated_cleaned[0:10].type(), current_generated_cleaned[0:10].size())
            # print(current_generated_cleaned[0:10])

            current_padding_mask = current_padding_mask.repeat(1, rollout_num)
            # print('rollout-current_padding_mask-: ', current_padding_mask.type(), current_padding_mask.size())
            # print(current_padding_mask)

            if rollout_padding_mask:
                rollout_padding_mask = torch.stack(rollout_padding_mask)
                # print('rollout-rollout_padding_mask-: ', rollout_padding_mask.type(), rollout_padding_mask.size())
                # print(rollout_padding_mask)
                current_padding_mask = torch.cat([current_padding_mask, rollout_padding_mask]).long()
            else:
                current_padding_mask = current_padding_mask.long()
            # print('rollout-current_padding_mask--: ', current_padding_mask.type(), current_padding_mask.size())
            # print(current_padding_mask)
            # assert False
            # comment_fake = torch.masked_select(current_generated, current_padding_mask)
            comment_fake = current_generated * current_padding_mask
            # print('rollout-comment_fake: ', comment_fake.type(), comment_fake.size())
            # print(comment_fake)

            # reward = disc(comment_fake)
            # print('rollout-reward: ', reward.type(), reward.size())
            # print(reward)
            # reward = torch.exp(reward[:, 1])
            # reward = reward.reshape(-1, rollout_num, batch_size)
            # print('rollout-reward-: ', reward.type(), reward.size())
            # print(reward)
            # reward = reward.sum(1)
            # print('rollout-reward--: ', reward.type(), reward.size())
            # print(reward)
            # reward = reward / 5
            # print('rollout-reward---: ', reward.type(), reward.size())
            # print(reward)

            comment_real = comment_batch[1:].repeat(1, rollout_num)
            # print('rollout-comment_real: ', comment_real.type(), comment_real.size())
            # print(comment_real)
            comment_fake_list, comment_real_list = comment_fake.t().tolist(), comment_batch[1:].repeat(1,
                                                                                                       rollout_num).t().tolist()

            reward2 = []
            for idx in range(len(comment_fake_list)):
                res = {0: [' '.join(dict_comment.getLabel(i) for i in
                                    utils.util.clean_up_sentence(comment_fake_list[idx], remove_unk=False,
                                                                 remove_eos=True))]}
                gts = {0: [' '.join(dict_comment.getLabel(i) for i in
                                    utils.util.clean_up_sentence(comment_real_list[idx], remove_unk=False,
                                                                 remove_eos=True))]}
                # res, gts = {k: [' '.join(str(i) for i in seq_list[k])] for k in range(len(seq_list))}, {k: [' '.join(str(i) for i in comment_batch_list[k])] for k in range(len(comment_batch_list))}
                # if idx < 4:
                #     print('forward_pg-res: ', res[0])
                #     print('forward_pg-gts: ', gts[0])
                score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
                # print('Bleu-1: ', np.mean(scores_Bleu[0]))
                # print('forward_pg-scores_Bleu: ', scores_Bleu)
                reward2.append(np.mean(scores_Bleu[0]))
            # reward, pred = sent_reward_func(seq.t().tolist(), comment_batch.t().tolist())
            # print('forward_pg-reward: ')
            # print(reward)
            reward2 = torch.Tensor(reward2)  # .repeat(seq_length, 1).cuda()
            # print('rollout-reward2: ', reward.type(), reward.size())
            # print(reward2)

            reward2 = reward2.reshape(rollout_num, batch_size)
            reward2 = reward2.sum(0).unsqueeze(0)
            # print('rollout-reward2-: ', reward2.type(), reward2.size())
            # print(reward2)
            reward2 = reward2 / 5
            # print('rollout-reward2--: ', reward2.type(), reward2.size())
            # print(reward2)

            return reward2

    def update(self, original_model):
        self.rnn = copy.deepcopy(original_model.rnn)
        # self.rnn.flatten_parameters()
        self.linear = copy.deepcopy(original_model.linear)
