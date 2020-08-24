#!/usr/bin/env bash

nohup python -m run.summarization.transformer_ast.train -l ruby > run/summarization/transformer_ast/ruby.log 2>&1
nohup python -m run.summarization.transformer_ast.train -l python > run/summarization/transformer_ast/python.log 2>&1
nohup python -m run.summarization.transformer_ast.train -l php > run/summarization/transformer_ast/php.log 2>&1
nohup python -m run.summarization.transformer_ast.train -l java > run/summarization/transformer_ast/java.log 2>&1
nohup python -m run.summarization.transformer_ast.train -l javascript > run/summarization/transformer_ast/javascript.log 2>&1
nohup python -m run.summarization.transformer_ast.train -l go > run/summarization/transformer_ast/go.log 2>&1
nohup python -m run.summarization.transformer_ast.train -l csharp > run/summarization/transformer_ast/csharp.log 2>&1