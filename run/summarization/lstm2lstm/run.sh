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