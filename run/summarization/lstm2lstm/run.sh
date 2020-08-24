#!/usr/bin/env bash

nohup python -m run.summarization.lstm2lstm.train -l ruby > run/summarization/lstm2lstm/ruby.log 2>&1
nohup python -m run.summarization.lstm2lstm.train -l python > run/summarization/lstm2lstm/python.log 2>&1
nohup python -m run.summarization.lstm2lstm.train -l php > run/summarization/lstm2lstm/php.log 2>&1
nohup python -m run.summarization.lstm2lstm.train -l java > run/summarization/lstm2lstm/java.log 2>&1
nohup python -m run.summarization.lstm2lstm.train -l javascript > run/summarization/lstm2lstm/javascript.log 2>&1
nohup python -m run.summarization.lstm2lstm.train -l go > run/summarization/lstm2lstm/go.log 2>&1
nohup python -m run.summarization.lstm2lstm.train -l csharp > run/summarization/lstm2lstm/csharp.log 2>&1

nohup python -m run.summarization.lstm2lstm.eval -l ruby > run/summarization/lstm2lstm/ruby.eval.log 2>&1
nohup python -m run.summarization.lstm2lstm.eval -l php > run/summarization/lstm2lstm/php.eval.log 2>&1
nohup python -m run.summarization.lstm2lstm.eval -l python > run/summarization/lstm2lstm/python.eval.log 2>&1
nohup python -m run.summarization.lstm2lstm.eval -l java > run/summarization/lstm2lstm/java.eval.log 2>&1
nohup python -m run.summarization.lstm2lstm.eval -l javascript > run/summarization/lstm2lstm/javascript.eval.log 2>&1
nohup python -m run.summarization.lstm2lstm.eval -l go > run/summarization/lstm2lstm/go.eval.log 2>&1
nohup python -m run.summarization.lstm2lstm.eval -l csharp > run/summarization/lstm2lstm/csharp.eval.log 2>&1