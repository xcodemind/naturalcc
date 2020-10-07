#!/usr/bin/env bash


nohup python -m run.summarization.lstm2lstm.train -l python_wan > run/summarization/lstm2lstm/python_wan.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.8 > run/summarization/lstm2lstm/python_wan0.8.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.8

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.7 > run/summarization/lstm2lstm/python_wan0.7.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.7

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.6 > run/summarization/lstm2lstm/python_wan0.6.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.6

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.5 > run/summarization/lstm2lstm/python_wan0.5.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.5

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.4 > run/summarization/lstm2lstm/python_wan0.4.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.4

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.3 > run/summarization/lstm2lstm/python_wan0.3.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.3

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.2 > run/summarization/lstm2lstm/python_wan0.2.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.2

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.1 > run/summarization/lstm2lstm/python_wan0.1.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.1

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.05 > run/summarization/lstm2lstm/python_wan0.05.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.05

nohup python -m run.summarization.lstm2lstm.train -l python_wan0.01 > run/summarization/lstm2lstm/python_wan0.01.log 2>&1
python -m run.summarization.lstm2lstm.eval -l python_wan0.01


# seq2seq
# python_wan
CUDA_VISIBLE_DEVICES=1,2 nohup python -m run.summarization.lstm2lstm.train -l python_wan > run/summarization/lstm2lstm/python_wan.log 2>&1 &
watch -n 2 "tail -n 10 run/summarization/lstm2lstm/python_wan.log"
CUDA_VISIBLE_DEVICES=1,2 python -m run.summarization.lstm2lstm.eval -l python_wan

# ruby
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.lstm2lstm.train -l ruby > run/summarization/lstm2lstm/ruby.log 2>&1 &
watch -n 2 "tail -n 10 run/summarization/lstm2lstm/ruby.log"
CUDA_VISIBLE_DEVICES=1,2,3 python -m run.summarization.lstm2lstm.eval -l ruby



nohup python -m run.summarization.lstm2lstm.train -l ruby.subtoken > run/summarization/lstm2lstm/ruby.subtoken 2>&1 &
watch -n 2 "tail -n 10 run/summarization/lstm2lstm/ruby.subtoken"

nohup python -m run.summarization.lstm2lstm.train -l python_wan.subtoken > run/summarization/lstm2lstm/python_wan.subtoken.log 2>&1 &
watch -n 2 "tail -n 10 run/summarization/lstm2lstm/python_wan.subtoken.log"

nohup python -m run.summarization.lstm2lstm.train -l python_wan.bpe > run/summarization/lstm2lstm/python_wan.bpe.log 2>&1 &
watch -n 2 "tail -n 10 run/summarization/lstm2lstm/python_wan.bpe.log"

nohup python -m run.summarization.lstm2lstm.train -l ruby.bpe > run/summarization/lstm2lstm/ruby.bpe.log 2>&1 &
watch -n 2 "tail -n 10 run/summarization/lstm2lstm/ruby.bpe.log"
python -m run.summarization.lstm2lstm.eval -l ruby.bpe