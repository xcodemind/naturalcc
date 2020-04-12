# Ruby
## Notes
需要简单调参数确定的：

- learning rate多少合适？特别是多模态的lr多少合适？
- clip_norm多少合适？
- rl_weight多少合适？
- h c fuse?
- leaf_path_k 多少合适？
- optim除了adam有必要用其他的吗？
- dropout多少合适？
- decoder_input_feed有没有效果？


注意事项：

- base中，dataset要加载全，而不是只有第一个json文件
- iterator范围为[1, 1 + len(training data)]
- 手动check生成的数据，注意头尾，及padding
- check metric计算，特别是mask的计算
- check loss计算，注意mask
- check tree-lstm准确性，看能否升级ddl简单加速下
- check attention和pointer
- Logger变成Info模式
- 参数设置：batch_size，lr，initialised weights，
- validate的时候可以取消rouge meteor，但记得测试的时候要加上
- 默认参数：batch_size: 128, lr: 0.001 
- init_weights刚开始训练的时候为none，在test或者resume的时候需要设置一下
- dataset_lng在切换语言的时候记得改一下

Summary:
- learning rate 0.0004比较合适，然后milestones[20, 40]，以gamma0.5衰减，没有warmup
- 多模态的引入更多信息，learning rate刚开始可以大一点
- enc_hc2dec_hc: 'h'
- max_grad_norm: -1取消clip_norm
- dropout: 0.2
- optic: Adam
- decoder_input_feed: False取消，为True的时候结果有点奇怪，可能存在一些bug
- leaf_path_k: 50


## Multi-Modal

### Tok (3.5min/epoch) 



### AST



### SBT


### Path



### MM-Tok8AST




### MM-Tok8Path



## +Attn
### Tok+Attn (2min/epoch) 

### SBT+Attn



### Path+Attn


### MM (TOk+AST)+Attn
#### Portion 1.0
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Mar-03-06-35-58.69dc064feeab8bbae7159124a5dae161.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization_100_small/mm2seq/ruby_p1.0_biTrue/train_sl/tok8ast-bs128-lr0.0004-attndot-pointerFalse-ttSLTrainer-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Mar-03-13-47-03.74900ba4801cc06222522839502bfaed.log
- server: rtx8000

|     B1 |     B2 |     B3 |     B4 |   Meteor |     R1 |     R2 |     R3 |     R4 |     RL |   Cider |
|--------|--------|--------|--------|----------|--------|--------|--------|--------|--------|---------|
| 0.1984 | 0.0486 | 0.0109 | 0.0034 |   0.0814 | 0.1831 | 0.0425 | 0.0089 | 0.0022 | 0.1707 |  0.2491 |

#### Portion 0.01 (1%)
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Mar-05-09-20-35.55d94217c8d7c06bc38f983898ea2ddc.log
- init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization_100_small/mm2seq/ruby_p0.01_biTrue/train_sl/tok8ast-bs128-lr0.0004-attndot-pointerFalse-ttSLTrainer-best-bleu1.pt'
-test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Mar-05-09-30-28.6e4206e74a6a0ce3f90d8c35cf806ff9.log

|     B1 |     B2 |     B3 |   B4 |   Meteor |     R1 |     R2 |   R3 |   R4 |     RL |   Cider |
|--------|--------|--------|------|----------|--------|--------|------|------|--------|---------|
| 0.1769 | 0.0096 | 0.0002 |    0 |   0.0538 | 0.1454 | 0.0084 |    0 |    0 | 0.1295 |  0.0453 |


### MM (Tok+Path)+Attn


## +Pointer
### Tok+Attn+Pointer (4min/epoch) 


### MM+Attn+Pointer


## +PG
### Tok

### MM



## +SC
### Tok



### MM

## +AC
### Tok

### Tok8Path



### Tok8AST
#### Portion 1.0
##### Pretrain-critic
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_CRITIC_2020-Mar-03-07-38-51.9f7c39d7139963e6bc40ec20eee46e8a.log



##### AC
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_AC_2020-Mar-03-10-44-04.246d44e03dd410537286b880e59f8400.log
- init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization_100_small/mm2seq/ruby_p1.0_biTrue/train_ac/tok8ast-bs128-lr0.0001-attndot-pointerFalse-ttACTrainer-ep7.pt'
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Mar-03-13-56-54.d54aad81350d5bcdc95768061f4d58bd.log
- server: rtx8000
|     B1 |     B2 |     B3 |     B4 |   Meteor |     R1 |     R2 |     R3 |    R4 |     RL |   Cider |
|--------|--------|--------|--------|----------|--------|--------|--------|-------|--------|---------|
| 0.2116 | 0.0562 | 0.0127 | 0.0044 |   0.0918 | 0.2232 | 0.0467 | 0.0103 | 0.002 | 0.2028 |  0.2826 |

#### Portion 0.01 (1%)
##### Pretrain-critic
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_CRITIC_2020-Mar-05-09-27-45.4143879c15a1ee5677abfc0bc3b725cf.log

##### AC
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_AC_2020-Mar-05-09-34-43.593c6242986915ca2fd8125c3a086d31.log
- init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization_100_small/mm2seq/ruby_p0.01_biTrue/train_ac/tok8ast-bs128-lr0.0001-attndot-pointerFalse-ttACTrainer-best-bleu1.pt'
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Mar-05-09-43-35.0e7e507780f217e86fefb7d82a0d4026.log

|     B1 |     B2 |   B3 |   B4 |   Meteor |     R1 |     R2 |   R3 |   R4 |     RL |   Cider |
|--------|--------|------|------|----------|--------|--------|------|------|--------|---------|
| 0.1758 | 0.0009 |    0 |    0 |   0.0478 | 0.1724 | 0.0002 |    0 |    0 | 0.1505 |  0.0623 |




