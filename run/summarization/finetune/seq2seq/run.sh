#!/usr/bin/env bash

# pretrain
nohup python -m run.summarization.finetune.seq2seq.pretrain > run/summarization/finetune/finetune/seq2seq/pretrain.log 2>&1
# finetune
nohup python -m run.summarization.finetune.seq2seq.finetune > run/summarization/finetune/finetune/seq2seq/finetune.log 2>&1