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
- batch_size 128, lr: 0.0004, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.bdc3c1fd51a1e65dbe5fffbd5b2a279e.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok-bs128-lr0.0004-attnNone-pointerFalse-ep20.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-15.c517e75003a67bcc6ea2c56956ebda3f.log

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1828 | 0.0341 | 0.0064 | 0.0023 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1552 |



#### Q1: h c fuse?

> A1: h的效果要好于hc，所以建议用h 


#### Q2: optim除了adam有必要用其他的吗？

> A2: optimiser还是选默认的Adam比较好


#### Q3: decoder_input_feed有没有效果？

> A3: decoder_input_feed=True，学习失败，可能是feed的时候有bug？



##### Q4: learning rate多少合适？特别是多模态的lr多少合适？

> A4: lr在4e-4 ~ 1e-3左右比较合适，lr_milestones: [10, 40]比较合适

#### Q5: clip_norm多少合适？

> A5: 目前还是不要clip norm好了（max_grad_norm＝－1），因为max_grad_norm设为1或者太大都没有效果，有机会可以试下max_grad_norm<1的情形。



#### Q6: dropout多少合适？

> A6: Dropout设置为0.2最合适，太大了不好


#### Q7: num_layers: 2? 


#### Q8: Bi-directional?


### AST
#### Q1: lr多少合适?
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: None
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-17-02-16-34.0234c6859a48bb724c4c9bbb7a61c95c.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/ast-bs128-lr0.0004-attnNone-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-17-10-58-43.b362fe99b92b67ea54724e66472088ff.log
- server: rtx8000

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1955 | 0.0467 | 0.0133 | 0.0055 |        0 |    0 |    0 |    0 |    0 |    0 |  0.2531 |


### SBT
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: None
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-16-15-08-02.c8aa2e02ad51f76efae8f2a8da51b662.log
- init_weights:
- test_log
- server: rtx8000

|     B1 |     B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|------|----------|------|------|------|------|------|---------|
| 0.1758 | 0.0176 | 0.0005 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0716 |


### Path

#### Q1: path采样多少条合适？
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: False, leaf_path_k: 50
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-16-22-42-21.f337b97b53c2e2ee01ae6ce1cee21d68.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/path-bs128-lr0.0004-attnNone-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-17-19-42-53.14b50b347fcf80a6a154d6b3c5d9d1e2.log
- server: 243

|    B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.188 | 0.0414 | 0.0099 | 0.0045 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1824 |

> A1: path采样长度k=50比较合适 (cider最大), leaf_path_k对于bleu的提升不明显，但是对于cider有显著提升。leaf_path_k=50，50次迭代，耗时1小时


#### Q2: lr多少合适？
> A2: path模态的learning rate待定


### MM-Tok8AST

- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: None
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-17-02-22-28.08b01ef22bb99bc840c5ad0138a4aeb5.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8ast-bs128-lr0.0004-attnNone-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-17-11-18-58.4179fc87622ea87d8efab72bf6ca2397.log
- server: rtx8000

|   B1 |     B2 |     B3 |    B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|------|--------|--------|-------|----------|------|------|------|------|------|---------|
| 0.18 | 0.0396 | 0.0116 | 0.005 |        0 |    0 |    0 |    0 |    0 |    0 |  0.2092 |



### MM-Tok8Path
#### Q1: lr多少合适？

> Q1: lr: 0.0004

#### Q2: code_modal_transform? (重新测一下best的model)
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15.df4d1e95efa79b8726024e730784604f.log
- init_weights:  
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-15-12-08-37.337431aeea368c0f19eaac172b52e6e8.log
- server: rtx8000

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1864 | 0.0436 | 0.0117 | 0.0063 |        0 |    0 |    0 |    0 |    0 |    0 |  0.2147 |

> A2: code_modal_transform为False效果比较好；即多模态直接concate就好了

## +Attn
### Tok+Attn (2min/epoch) 
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], attn_type: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15-20-31-23.c423d730941270d02327ca6907103651.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/attnDot/tok-bs128-lr0.0004-attndot-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-11-10-52.869ef7c9084538b766e015cb002102f0.log

|     B1 |    B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|-------|--------|--------|----------|------|------|------|------|------|---------|
| 0.2073 | 0.059 | 0.0175 | 0.0066 |        0 |    0 |    0 |    0 |    0 |    0 |  0.3431 |

- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/attnDot/tok-bs128-lr0.0004-attndot-pointerFalse-best-cider.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-11-07-18.f1a7caec9f7b42210407855eecb91304.log
- server: 243

|     B1 |     B2 |    B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|-------|--------|----------|------|------|------|------|------|---------|
| 0.2055 | 0.0583 | 0.017 | 0.0072 |        0 |    0 |    0 |    0 |    0 |    0 |  0.3579 |


### SBT+Attn
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-17-02-31-56.f0a644ea28815dcda72f638eab406a1b.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok-bs128-lr0.0004-attndot-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-17-11-31-29.b78f050ffe43a4251841a10eacadc915.log
- server:  rtx8000

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1875 | 0.0408 | 0.0103 | 0.0022 |        0 |    0 |    0 |    0 |    0 |    0 |  0.2153 |


### Path+Attn
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-16-22-47-17.c7248663ba4c8c89c0a39e7efdcbbbad.log
- init_weights:
- test_log
- server: 243 (running...)



### MM (TOk+AST)+Attn
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-17-02-27-34.4fb4edd1bd5b6ec47a7dbcfd134fa7f1.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8ast-bs128-lr0.0004-attndot-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-17-11-24-16.dae07b83c3d9ef52c0f46fcc11073e6d.log
- server: rtx8000

|    B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.211 | 0.0614 | 0.0181 | 0.0068 |        0 |    0 |    0 |    0 |    0 |    0 |  0.3661 |


### MM (Tok+Path)+Attn
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False, attn_type: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15-20-39-52.54960dce5eefe20c56e5a43878fdd26b.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path_attnDot/tok8path-bs128-lr0.0004-attndot-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-11-13-02.07ce0b17a6b3b400f15204247268975a.log
- server: new-240

|     B1 |     B2 |    B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|-------|--------|----------|------|------|------|------|------|---------|
| 0.2161 | 0.0631 | 0.015 | 0.0031 |        0 |    0 |    0 |    0 |    0 |    0 |  0.3679 |

- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path_attnDot/tok8path-bs128-lr0.0004-attndot-pointerFalse-best-cider.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-11-16-56.4720681bc00f1e33c0f4e4cd630fb829.log

|     B1 |     B2 |    B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|-------|--------|----------|------|------|------|------|------|---------|
| 0.2079 | 0.0606 | 0.015 | 0.0051 |        0 |    0 |    0 |    0 |    0 |    0 |  0.3747 |


## +Pointer
### Tok+Attn+Pointer (4min/epoch) 
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], attn_type: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15-20-38-05.296dd3bb1181d3a7b2c92ee0f779e544.log
- init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/pointerTrue/tok-bs128-lr0.0004-attndot-pointerTrue-best-bleu1.pt' 
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-11-04-03.074266d753492b7f0672e0bb9ee8e696.log
- server: 243

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.2388 | 0.0832 | 0.0229 | 0.0066 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5307 |

- init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/pointerTrue/tok-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt' 
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-11-25-50.ddf154844a077227985297c8b6f3fdea.log

|     B1 |     B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|------|----------|------|------|------|------|------|---------|
| 0.2351 | 0.0836 | 0.0261 | 0.01 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5836 |


### MM+Attn+Pointer
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False, attn_type: dot
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15-20-45-06.d9d79faa0bfc81b50831166d994047c6.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path_pointerTrue/tok8path-bs128-lr0.0004-attndot-pointerTrue-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-10-58-52.b1e28a3d0b7841eb307bce778ea5111a.log
- server: new-240

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.2389 | 0.0842 | 0.0248 | 0.0092 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5619 |


- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path_pointerTrue/tok8path-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-11-21-30.5b69ed7c34507f1927d3a729ea3d8dde.log

|    B1 |     B2 |     B3 |    B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|-------|----------|------|------|------|------|------|---------|
| 0.237 | 0.0863 | 0.0293 | 0.012 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5882 |


## +PG
### Tok
- batch_size 128, lr_rl: 0.0001
- pre-trained init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/pointerTrue/tok-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt
- train_log: 
- init_weights: 
- test_log: 


### MM
- batch_size 128, lr_rl: 0.0001
- pre-trained init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path_pointerTrue/tok8path-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt
- train_log: 
- init_weights: 
- test_log: 


## +SC
### Tok
##### Q1: lr_rl多少合适
- batch_size 128, lr_rl: 0.0001
- pre-trained init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/pointerTrue/tok-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SC_2020-Feb-16-17-25-35.d8957aad7bc5b5aa7dea14e1df649eaf.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sc/tok-bs128-lr0.0001-attndot-pointerTrue-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-22-30-15.08f3ac66bb0e61b0d24ca603dd6bab50.log
- serve: 243

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.2572 | 0.0827 | 0.0206 | 0.0067 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5387 |

- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sc/tok-bs128-lr0.0001-attndot-pointerTrue-best-cider.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-22-33-29.c6f4b1eff76931d1e21ead18bc844c04.log

|     B1 |    B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|-------|--------|--------|----------|------|------|------|------|------|---------|
| 0.2004 | 0.066 | 0.0215 | 0.0091 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5755 |


- batch_size 128, lr_rl: 0.00005
- pre-trained init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/pointerTrue/tok-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt
- init_weights: 
- train_log: 
- test_log: 
- serve: 243


- batch_size 128, lr_rl: 0.0002
- pre-trained init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/pointerTrue/tok-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt
- init_weights: 
- train_log: 
- test_log: 
- serve: 243


### MM
##### Q1: lr_rl多少合适
- batch_size 128, lr_rl: 0.0001
- pre-trained init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path_pointerTrue/tok8path-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SC_2020-Feb-16-17-15-32.b34dcc3adb478f9fcc2b26e1b09f03b9.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sc/tok8path-bs128-lr0.0001-attndot-pointerTrue-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-21-20-13.5bfde8572fd413a83721cd7aec92164e.log
- server: 243

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.2451 | 0.0745 | 0.0175 | 0.0053 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5386 |

- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sc/tok8path-bs128-lr0.0001-attndot-pointerTrue-best-cider.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-16-21-29-12.caa968233f7e6522a1cb9980f150d240.log

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1939 | 0.0567 | 0.0184 | 0.0073 |        0 |    0 |    0 |    0 |    0 |    0 |  0.5963 |

## +AC
### Tok
#### Pretrain-Critic***** 在跑
- batch_size 128, lr_critic: 0.0001
- pre-trained init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/pointerTrue/tok-bs128-lr0.0004-attndot-pointerTrue-best-cider.pt'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_CRITIC_2020-Feb-16-20-06-15.dbd9c4c739bbb226cd0256b735123982.log

#### Train AC
- batch_size 128, lr_rl: 0.0001, lr_critic: 0.0001
- init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt'
- init_weights_critic: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_critic/critic-tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt'
- train_log: /data/wanyao/Dropbox/ghproj-titan/naturalcodev2/run/summarization/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_AC_2020-Feb-05.0972d81aa593dc0e32cf7b3c97bf9c46.log
- test_log: /data/wanyao/Dropbox/ghproj-titan/naturalcodev2/run/summarization/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-05.d59e6a71921fb0136db87145d5f64aa8.log

|     B1 |     B2 |     B3 |     B4 |   Meteor |     R1 |     R2 |     R3 |     R4 |    RL |   Cider |
|--------|--------|--------|--------|----------|--------|--------|--------|--------|-------|---------|
| 0.1905 | 0.0266 | 0.0021 | 0.0009 |        0 | 0.1696 | 0.0227 | 0.0018 | 0.0007 | 0.156 |  0.1015 |


### Tok8AST ****
#### Pretrain-Critic

#### Train AC
