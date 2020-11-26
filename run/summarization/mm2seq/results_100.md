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
- <font color=red>Base: </font>
- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.0', decoder_input_feed: 'False'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-13.fc6c5cd6c5da96955c1aa276afb2f947.log
- init_weights: '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt' 
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.509e027060d749388691c3af4a417914.log

|    B1 |     B2 |     B3 |    B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|-------|----------|------|------|------|------|------|---------|
| 0.175 | 0.0308 | 0.0086 | 0.005 |        0 |    0 |    0 |    0 |    0 |    0 |    0.18 |


- <font color=red>Optimal: </font>
- batch_size 128, lr: 0.0004, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.bdc3c1fd51a1e65dbe5fffbd5b2a279e.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok-bs128-lr0.0004-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-15.e5552c8a6256ff2b5a3346c84539d687.log

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1764 | 0.0339 | 0.0075 | 0.0033 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1598 |

- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok-bs128-lr0.0004-attnNone-pointerFalse-ep20.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-15.c517e75003a67bcc6ea2c56956ebda3f.log

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1828 | 0.0341 | 0.0064 | 0.0023 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1552 |



#### Q1: h c fuse?
- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], enc_hc2dec_hc: 'hc'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-13.817c70475c2866f8afa1d76e0579775c.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/enc_hc2dec_hc/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.efbab3d6af30588344f97fb236969b9f.log

|     B1 |     B2 |    B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|-------|--------|----------|------|------|------|------|------|---------|
| 0.1707 | 0.0294 | 0.007 | 0.0029 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1688 |


> A1: h的效果要好于hc，所以建议用h 


#### Q2: optim除了adam有必要用其他的吗？
- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], optim: 'AdamW'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-13.4a80e15afce2a5bce50d369aa1ee5ee6.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/optim_AdamW/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.6589b8176690d0764d5e5067ecdb3a88.log

|     B1 |     B2 |     B3 |    B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|-------|----------|------|------|------|------|------|---------|
| 0.1698 | 0.0277 | 0.0064 | 0.003 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1534 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], optim: 'Adagrad'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-13.068cfe9105d85574a4eb57a18412e16e.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/optim_Adagrad/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.533f21778cdf8310372e208ace6c8ffd.log

|     B1 |     B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|------|----------|------|------|------|------|------|---------|
| 0.1577 | 0.0149 | 0.0002 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0451 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], optim: 'RMSprop'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-13.249e9d2ac5e994ef7d24bcd01c3bd398.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/optim_RMSprop/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.3c4673917c2ec92bb8b3e2a593414be2.log

|     B1 |     B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|------|----------|------|------|------|------|------|---------|
| 0.1492 | 0.0132 | 0.0001 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0315 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], optim: 'SGD'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-13.d9a34781cacf2cca11bbe5063d4adaf7.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/optim_SGD/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.965bf1a2b25b78386d9559ee1f0cc64a.log

|     B1 |   B2 |   B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|------|------|------|----------|------|------|------|------|------|---------|
| 0.0046 |    0 |    0 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |       0 |

> A2: optimiser还是选默认的Adam比较好


#### Q3: decoder_input_feed有没有效果？
- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], decoder_input_feed: True
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.f9fae7e1ae32078771e8829c1305d882.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/decoder_input_feed_True/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.b76c7eb97bec7f2c4f95d37624dd1dd4.log
- server: amax-xp-24322

|     B1 |     B2 |   B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|------|------|----------|------|------|------|------|------|---------|
| 0.0316 | 0.0005 |    0 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |       0 |

> A3: decoder_input_feed=True，学习失败，可能是feed的时候有bug？



##### Q4: learning rate多少合适？特别是多模态的lr多少合适？
- batch_size 128, lr: 0.01, lr_gamma: 0.1, lr_milestones: [20, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.18ff77bb651bb2bcee724c9b3060d1bf.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/lr_0.01/tok-bs128-lr0.01-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.a329a7ee32c9c1aeb943e40dd66bb9ed.log

|     B1 |     B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|------|----------|------|------|------|------|------|---------|
| 0.1492 | 0.0132 | 0.0001 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0315 |


- batch_size 128, lr: 0.0001, lr_gamma: 0.1, lr_milestones: [20, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.a27e290f11112e379be1c62d04e20495.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/lr_0.0001/tok-bs128-lr0.0001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.c5881dc9ffbff5a258204cb442de97ce.log

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1526 | 0.0142 | 0.0007 | 0.0001 |        0 |    0 |    0 |    0 |    0 |    0 |   0.039 |


- <font color=red>batch_size 128, lr: 0.0004, lr_gamma: 0.1, lr_milestones: [20, 40]</font>
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.9bb2664a5c79219f961b23123a18f65d.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/lr_0.0004/tok-bs128-lr0.0004-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.78c5d1b7a3df6e48730d3b1885743240.log

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1802 | 0.0362 | 0.0075 | 0.0034 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1768 |



- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [10, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.b36cb05fe4d87639b929c2b8275cee89.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/lr_0.001_milestones1040/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.9e75a39c0fe821c7148ff307c6ae7d52.log

|     B1 |     B2 |     B3 |    B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|-------|----------|------|------|------|------|------|---------|
| 0.1789 | 0.0313 | 0.0068 | 0.003 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1532 |

> A4: lr在4e-4 ~ 1e-3左右比较合适，lr_milestones: [10, 40]比较合适

#### Q5: clip_norm多少合适？
- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: 1.0
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.5fceb2f18e79622ef2a4a9c8ffb5807c.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/max_grad_norm1.0/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.b7bca1d30c269bd18369b4efdd899b18.log
- server: amax-xp-24322

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1736 | 0.0303 | 0.0087 | 0.0049 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1724 |



- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: 10.0
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.0918e747065b3900ad515e235f4dacec.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/max_grad_norm10/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.967427cefac27d69256cb437ed6cb3f4.log
- server: amax-xp-24322

|    B1 |     B2 |     B3 |    B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|-------|----------|------|------|------|------|------|---------|
| 0.175 | 0.0308 | 0.0086 | 0.005 |        0 |    0 |    0 |    0 |    0 |    0 |    0.18 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: 20.0
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.81a50d34266a09084c5a0a6eba4f7e38.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/max_grad_norm20/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.f83901a06cbe9ec08da4e737abc1468a.log
- server: amax-new-24022

|    B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.175 | 0.0347 | 0.0092 | 0.0051 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1806 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: 100.0
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.ce97cd72513064d164bca45281109ff2.log
- init_weights: 
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.2489a15829b4ab529e158fced6e3e269.log
- server: amax-new-24022

|    B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.175 | 0.0347 | 0.0092 | 0.0051 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1806 |


> A5: 目前还是不要clip norm好了（max_grad_norm＝－1），因为max_grad_norm设为1或者太大都没有效果，有机会可以试下max_grad_norm<1的情形。



#### Q6: dropout多少合适？
- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], dropout: '0.2'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.039152a0cb807ae4e0a50b8b939781ff.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/dropout_0.2/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.c64d409abd8a8076ba05f4114d2aafb1.log
- server: amax-xp-24322

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1787 | 0.0355 | 0.0083 | 0.0041 |        0 |    0 |    0 |    0 |    0 |    0 |   0.176 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], dropout: '0.4'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.24ef94c0d5aa2229f909b0b264ae1631.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/dropout_0.4/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.39b8a0cf7e5a6b431de91ffcf5a2574e.log
- server: amax-xp-24322

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1745 | 0.0344 | 0.0092 | 0.0037 |        0 |    0 |    0 |    0 |    0 |    0 |   0.177 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], dropout: '0.6'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.460035d900866aa11263c209fc2cfa43.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/dropout_0.6/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.a18c233df97a72b8937f3fe96002df33.log
- server: amax-new-24022

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1645 | 0.0281 | 0.0061 | 0.0027 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1385 |


- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40], dropout: '0.8'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.e63718e8f7baf55d6fbb6012489a5d15.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/dropout_0.8/tok-bs128-lr0.001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-14.d14d5427ce6b8b4328aaf17069a23a58.log
- server: amax-new-24022

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1442 | 0.0159 | 0.0022 | 0.0007 |        0 |    0 |    0 |    0 |    0 |    0 |   0.074 |


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
leaf_path_k对于bleu的提升不明显，但是对于cider有显著提升。leaf_path_k=50，50次迭代，耗时1小时

path(10)

|     B1 |     B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|------|----------|------|------|------|------|------|---------|
| 0.1722 | 0.0171 | 0.0005 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0691 |

path(20)

|     B1 |     B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|------|----------|------|------|------|------|------|---------|
| 0.1739 | 0.0173 | 0.0005 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0708 |

path(30)

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1767 | 0.0256 | 0.0052 | 0.0029 |        0 |    0 |    0 |    0 |    0 |    0 |   0.102 |

path(40)

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1757 | 0.0258 | 0.0054 | 0.0021 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0992 |

path(50)

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1748 | 0.0279 | 0.0053 | 0.0021 |        0 |    0 |    0 |    0 |    0 |    0 |    0.12 |


- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False', attn: False, leaf_path_k: 50
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-16-22-42-21.f337b97b53c2e2ee01ae6ce1cee21d68.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/path-bs128-lr0.0004-attnNone-pointerFalse-best-bleu1.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-17-19-42-53.14b50b347fcf80a6a154d6b3c5d9d1e2.log
- server: 243

|    B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.188 | 0.0414 | 0.0099 | 0.0045 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1824 |



> A1: path采样长度k=50比较合适 (cider最大)


#### Q2: lr多少合适？
- path模态的learning rate待定

> A2: 


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

- batch_size 128, lr: 0.0004, lr_gamma: 0.1, lr_milestones: [20, 40], max_grad_norm: -1, enc_hc2dec_hc: 'h', optim: 'Adam', dropout: '0.2', decoder_input_feed: 'False'
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.88893c0a0f8d2728b67828f7e00e460f.log
- init_weights: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path-bs128-lr0.0004-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-15.ffc6af3954e900fea9b308ace7c938e1.log
- server: amax-xp-24322

|    B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|-------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.168 | 0.0305 | 0.0063 | 0.0023 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1382 |


#### Q1: 多模态lr多少合适？
- batch_size 128, lr: 0.001, lr_gamma: 0.1, lr_milestones: [20, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.11b7ef6a2b7a763630ce84e2920cb202.log
- init_weights:  
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-15.e5e8b98e1412b5169f7613dbba81274e.log
- server: amax-new-24022

|     B1 |    B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|-------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1733 | 0.031 | 0.0058 | 0.0026 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1347 |


- batch_size 128, lr: 0.0001, lr_gamma: 0.1, lr_milestones: [20, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-14.2acf32379a3e060b04eb84a1fa3cba9e.log
- init_weights:  /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/ruby/train_sl/tok8path_lr_0.0001/tok8path-bs128-lr0.0001-attnNone-pointerFalse-ep50.pt
- test_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TEST_2020-Feb-15.407a72b628e1dab63fd4a2f153c693c8.log
- server: amax-new-24022

|     B1 |     B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1636 | 0.0218 | 0.0023 | 0.0003 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0829 |

----

- batch_size 128, lr: 0.01, lr_gamma: 0.1, lr_milestones: [10, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15.0dceb32b6928af7f4a73402ec30d0c03.log
- init_weights:  
- test_log: 
- server: amax-titan-24322
- 变差 0.16左右



- batch_size 128, lr: 0.001, lr_gamma: 0.5, lr_milestones: [10, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15.b711b1448e0980a3bef7788be735a20a.log
- init_weights:  
- test_log: 
- server: amax-titan-24322
- 变差 0.16左右

- batch_size 128, lr: 0.001, lr_gamma: 0.5, lr_milestones: [10, 20, 30, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15.92e9b9a34af77169f40313158cc18b03.log
- init_weights:  
- test_log: 
- server: amax-titan-24322
- 变差 0.16左右

---


- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40]
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_UNILANG_MM2SEQ_TRAIN_SL_2020-Feb-15.fe151edbe3afe3b07d1a6d715ddf402d.log
- init_weights:  
- test_log: 
- server: rtx8000
- 变差0.17左右

> A1: 加入多模态后，引入信息更多，刚开始lr可以大一点，不然学得太慢，50个epoch可能还没学好，效果变差。

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


### Tok8Path ****
#### Pretrain-Critic

#### Train AC



### Tok8AST
#### Pretrain-Critic

#### Train AC


## DTRL time:baseline*2
python+ruby

|     B1 |    B2 |     B3 |   B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|-------|--------|------|----------|------|------|------|------|------|---------|
| 0.1726 | 0.016 | 0.0005 |    0 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0606 |

## MAML
10

|     B1 |    B2 |     B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|-------|--------|--------|----------|------|------|------|------|------|---------|
| 0.1767 | 0.023 | 0.0019 | 0.0001 |        0 |    0 |    0 |    0 |    0 |    0 |  0.0879 |


all

|     B1 |     B2 |    B3 |     B4 |   Meteor |   R1 |   R2 |   R3 |   R4 |   RL |   Cider |
|--------|--------|-------|--------|----------|------|------|------|------|------|---------|
| 0.1786 | 0.0329 | 0.007 | 0.0029 |        0 |    0 |    0 |    0 |    0 |    0 |  0.1438 |
