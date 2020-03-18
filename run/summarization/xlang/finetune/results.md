# Fine-Tuning
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



## Python -> Ruby

#### Pre-training
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False, attn_type: dot, pointer: True
- train_log: 
- server: rtx8000

#### Fine-tuning
- train_log: 
- init_weights: xxx
- test_log: 
- server: 

## Java -> Ruby

#### Pre-training
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False, attn_type: dot, pointer: True
- train_log: 
- server: 

#### Fine-tuning
- train_log: 
- init_weights: xxx
- test_log: 
- server: 


## Javascript -> Ruby

#### Pre-training
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False, attn_type: dot, pointer: True
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_XLANG_FINETUNE_PRETRAIN_2020-Feb-18-21-47-15.ad213fe6d179adc1c3a06b030eb7284c.log
- server: 243


#### Fine-tuning
- train_log: 
- init_weights: xxx
- test_log: 
- server: 


### PHP -> Ruby

#### Pre-training
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False, attn_type: dot, pointer: True
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_XLANG_FINETUNE_PRETRAIN_2020-Feb-18-21-43-22.d4d33a259c46f48bccc2aa2214f9744d.log
- server: 243

#### Fine-tuning
- train_log: 
- init_weights: xxx
- test_log: 
- server: 



### Go -> Ruby

#### Pre-training
- batch_size 128, lr: 0.0004, lr_gamma: 0.5, lr_milestones: [20, 40], code_modal_transform: False, attn_type: dot, pointer: True
- train_log: /data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/log/SUMMARIZATION_XLANG_FINETUNE_PRETRAIN_2020-Feb-18-22-11-57.2b6b7d726a28ac3a228ff7e7887b82c0.log
- server: 243

#### Fine-tuning
- train_log: 
- init_weights: xxx
- test_log: 
- server: 

