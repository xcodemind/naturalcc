#!/usr/bin/env bash

nohup python -m run.summarization.travtrans.train -l ruby > run/summarization/travtrans/ruby.log 2>&1
nohup python -m run.summarization.travtrans.train -l python > run/summarization/travtrans/python.log 2>&1
nohup python -m run.summarization.travtrans.train -l php > run/summarization/travtrans/php.log 2>&1
nohup python -m run.summarization.travtrans.train -l java > run/summarization/travtrans/java.log 2>&1
nohup python -m run.summarization.travtrans.train -l javascript > run/summarization/travtrans/javascript.log 2>&1
nohup python -m run.summarization.travtrans.train -l go > run/summarization/travtrans/go.log 2>&1
nohup python -m run.summarization.travtrans.train -l csharp > run/summarization/travtrans/csharp.log 2>&1


nohup python -m run.summarization.travtrans.eval -l ruby > run/summarization/travtrans/ruby.eval.log 2>&1
nohup python -m run.summarization.travtrans.eval -l php > run/summarization/travtrans/php.eval.log 2>&1
nohup python -m run.summarization.travtrans.eval -l python > run/summarization/travtrans/python.eval.log 2>&1
nohup python -m run.summarization.travtrans.eval -l java > run/summarization/travtrans/java.eval.log 2>&1
nohup python -m run.summarization.travtrans.eval -l javascript > run/summarization/travtrans/javascript.eval.log 2>&1
nohup python -m run.summarization.travtrans.eval -l go > run/summarization/travtrans/go.eval.log 2>&1
nohup python -m run.summarization.travtrans.eval -l csharp > run/summarization/travtrans/csharp.eval.log 2>&1