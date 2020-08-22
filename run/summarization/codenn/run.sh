#!/usr/bin/env bash

nohup python -m run.summarization.codenn.train -l ruby > run/summarization/codenn/ruby.log 2>&1
nohup python -m run.summarization.codenn.train -l php > run/summarization/codenn/php.log 2>&1
nohup python -m run.summarization.codenn.train -l python > run/summarization/codenn/python.log 2>&1
nohup python -m run.summarization.codenn.train -l java > run/summarization/codenn/java.log 2>&1
nohup python -m run.summarization.codenn.train -l javascript > run/summarization/codenn/javascript.log 2>&1
nohup python -m run.summarization.codenn.train -l go > run/summarization/codenn/go.log 2>&1
nohup python -m run.summarization.codenn.train -l csharp > run/summarization/codenn/csharp.log 2>&1
