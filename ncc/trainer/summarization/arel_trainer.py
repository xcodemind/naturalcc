# -*- coding: utf-8 -*-
# ref: https://github.com/eric-xw/AREL
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from ncc import LOGGER
from ncc.trainer import *
from ncc.model import *
from ncc.model.template import *
from ncc.dataset import *
from ncc.metric import *
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.util_eval import *
from ncc.utils.util_gan import AlterFlag

class ARELTrainer(Trainer):
    '''
    Adversarial Reward Learning Trainer
    '''

    def __init__(self, config: Dict, ) -> None:
        super(ARELTrainer, self).__init__(config)

    def train(self, model: IModel, disc: IModel, dataset: UnilangDataloader, pg_criterion, lm_criterion: BaseLoss,
              disc_criterion, optimizer: Optimizer, disc_optimizer: Optimizer, SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time
        alter_flag = AlterFlag(D_iters=model.config['arel']['D_iter'], G_iters=model.config['arel']['G_iter'],
                               always=model.config['arel']['always'])

        for epoch in range(1, 1 + model.config['arel']['train_epoch_arel']):
            model.train(), disc.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                # if model.config['training']['pointer']:
                #     code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
                # else:
                #     code_oovs = None
                comment_target = batch['comment'][2][:, :model.config['training']['max_predict_length']]

                LOGGER.info('alter_flag.flag: {}'.format(alter_flag.flag))
                if alter_flag.flag == 'disc':
                    enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
                    sample_opt = {'sample_max': 0, 'seq_length': model.config['training']['max_predict_length']}
                    # seq, seq_logprobs, seq_logp_gathered, seq_padding_mask, seq_lprob_sum, dec_output, dec_hidden, \
                    #     = model.decoder.forward(batch, enc_output, dec_hidden, enc_mask, sample_opt)
                    seq, seq_logprobs, seq_logp_gathered, seq_lprob_sum, comment_target_padded, = \
                        model.decoder.sample(batch, enc_output, dec_hidden, enc_mask, sample_opt)
                    # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
                    # comment_critic, comment_logprobs_critic, comment_logp_gathered_critic, comment_padding_mask_critic, comment_lprob_sum, \
                    # dec_output_critic, dec_hidden_critic, = self.decoder.forward(batch, enc_output_critic, dec_hidden_critic,
                    #                                                              enc_mask_critic, sample_opt)
                    print('seq: ', seq.size())
                    print(seq)

                    gen_score = disc.forward(batch, seq)
                    gt_score = disc.forward(batch, comment_target)
                    disc_loss = -torch.sum(gt_score) + torch.sum(gen_score)
                    print('Epoch: %s, iter: %s, dis_loss: %s' % (epoch, iteration, disc_loss.item()))
                    avg_pos_score = torch.mean(gt_score)
                    avg_neg_score = torch.mean(gen_score)
                    LOGGER.info("pos reward {} neg reward {}".format(avg_pos_score, avg_neg_score))
                    disc_loss.backward()
                    nn.utils.clip_grad_norm(disc.parameters(), disc.config['arel']['grad_clip'], norm_type=2)
                    disc_optimizer.step()
                    assert False
                elif alter_flag.flag == 'gen':
                    comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
                    enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
                    sample_opt = {'sample_max': 0, 'seq_length': critic.config['max_predict_length']}
                    # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
                    # _, _, _, _, _, _, dec_hidden_output, dec_hidden_critic, \
                    #     = critic.decoder.forward_pg(batch, enc_output_critic, dec_hidden_critic, enc_mask_critic, dataset.token_dicts, sample_opt)
                    seq, seq_logprobs, seq_logp_gathered, seq_lprob_sum, comment_target_padded, = model.decoder.forward(batch, enc_output, dec_hidden, enc_mask, sample_opt)
                    # print('critic: ', critic)

                    gen_score = disc(batch, seq)
                    print('gen_score: ', type(gen_score), gen_score.size())
                    print(gen_score)
                    # print('normed_seq_log_probs: ', type(normed_seq_log_probs), normed_seq_log_probs.size())
                    # print(normed_seq_log_probs)
                    rewards = gen_score - 0.001 * normed_seq_log_probs
                    print('rewards: ', type(rewards), rewards.size())
                    print(rewards)
                    # with open("/tmp/reward.txt", "a") as f:
                    #    print(" ".join(map(str, rewards.data.cpu().numpy())), file=f)
                    gen_loss = pg_criterion(seq_logprobs, seq, comment_padding_mask, reward)  # -value2
                    print('gen_loss: ', gen_loss)
                    # if logger.iteration % opt.losses_log_every == 0:
                    avg_pos_score = torch.mean(gen_score)
                    logging.info(
                        "average reward: {} average IRL score: {}".format(avg_score, avg_pos_score))

                    tf_loss = crit(model(feature_fc, target), target)
                    print("rl_loss / tf_loss = ", loss.item() / tf_loss.item())
                    loss = opt.rl_weight * loss + (1 - opt.rl_weight) * tf_loss
                    loss.backward()
                    nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip, norm_type=2)
                    optimizer.step()

                if iteration % model.config['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; time: {}'.format(
                        epoch, model.config['all_epoch'], iteration, len(dataset['train']), total_loss / iteration,
                        str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            if epoch <= model.config['arel']['train_epoch_arel']:
                if SAVE_DIR is not None:
                    model_name = '{}-bs{}-lr{}-attn{}-pointer{}-ep{}'.format('8'.join(model.config['code_modalities']),
                                                                                    model.config['batch_size'],
                                                                                    model.config['rl']['lr_critic'],
                                                                                    model.config['attn_type'],
                                                                                    model.config['pointer'], epoch)
                    model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                    torch.save(model.state_dict(), model_path)
                    LOGGER.info('Dumping arel model in {}'.format(model_path))
                # Evaluator.summarization_eval(critic, dataset['valid'], dataset.token_dicts, )
            else:
                pass
        LOGGER.info('{} train end'.format(self))
