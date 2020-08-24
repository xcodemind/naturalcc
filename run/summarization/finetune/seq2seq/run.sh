#!/usr/bin/env bash

# pretrain
nohup python -m run.summarization.finetune.seq2seq.pretrain > run/summarization/finetune/seq2seq/pretrain.csn.log 2>&1
# finetune
nohup python -m run.summarization.finetune.seq2seq.finetune > run/summarization/finetune/seq2seq/finetune.csn2csharp.log 2>&1