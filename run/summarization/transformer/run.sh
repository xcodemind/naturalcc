#!/usr/bin/env bash

nohup python -m run.summarization.transformer.train -l ruby > run/summarization/transformer/ruby.log 2>&1
nohup python -m run.summarization.transformer.train -l python > run/summarization/transformer/python.log 2>&1
nohup python -m run.summarization.transformer.train -l php > run/summarization/transformer/php.log 2>&1
nohup python -m run.summarization.transformer.train -l java > run/summarization/transformer/java.log 2>&1
nohup python -m run.summarization.transformer.train -l javascript > run/summarization/transformer/javascript.log 2>&1
nohup python -m run.summarization.transformer.train -l go > run/summarization/transformer/go.log 2>&1
nohup python -m run.summarization.transformer.train -l csharp > run/summarization/transformer/csharp.log 2>&1

nohup python -m run.summarization.transformer.eval -l ruby > run/summarization/transformer/ruby.eval.log 2>&1
nohup python -m run.summarization.transformer.eval -l php > run/summarization/transformer/php.eval.log 2>&1
nohup python -m run.summarization.transformer.eval -l python > run/summarization/transformer/python.eval.log 2>&1
nohup python -m run.summarization.transformer.eval -l java > run/summarization/transformer/java.eval.log 2>&1
nohup python -m run.summarization.transformer.eval -l javascript > run/summarization/transformer/javascript.eval.log 2>&1
nohup python -m run.summarization.transformer.eval -l go > run/summarization/transformer/go.eval.log 2>&1
nohup python -m run.summarization.transformer.eval -l csharp > run/summarization/transformer/csharp.eval.log 2>&1



nohup python -m run.summarization.transformer.train -l csharp-bpe > run/summarization/transformer/csharp-bpe.log 2>&1
