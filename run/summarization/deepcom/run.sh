#!/usr/bin/env bash

nohup python -m run.summarization.deepcom.train -l ruby > run/summarization/deepcom/ruby.log 2>&1
nohup python -m run.summarization.deepcom.train -l php > run/summarization/deepcom/php.log 2>&1
nohup python -m run.summarization.deepcom.train -l python > run/summarization/deepcom/python.log 2>&1
nohup python -m run.summarization.deepcom.train -l java > run/summarization/deepcom/java.log 2>&1
nohup python -m run.summarization.deepcom.train -l javascript > run/summarization/deepcom/javascript.log 2>&1
nohup python -m run.summarization.deepcom.train -l go > run/summarization/deepcom/go.log 2>&1
nohup python -m run.summarization.deepcom.train -l csharp > run/summarization/deepcom/csharp.log 2>&1
