#!/usr/bin/env bash

# finetune
nohup python -m run.summarization.finetune.python_wan.finetune -l ruby > run/summarization/finetune/python_wan/finetune.ruby2python.log 2>&1
nohup python -m run.summarization.finetune.python_wan.finetune -l php > run/summarization/finetune/python_wan/finetune.php2python.log 2>&1
nohup python -m run.summarization.finetune.python_wan.finetune -l java > run/summarization/finetune/python_wan/finetune.java2python.log 2>&1
nohup python -m run.summarization.finetune.python_wan.finetune -l javascript > run/summarization/finetune/python_wan/finetune.javascript2python.log 2>&1
nohup python -m run.summarization.finetune.python_wan.finetune -l go > run/summarization/finetune/python_wan/finetune.go2python.log 2>&1
nohup python -m run.summarization.finetune.python_wan.finetune -l csn > run/summarization/finetune/python_wan/finetune.csn2python.log 2>&1


python -m run.summarization.finetune.python_wan.eval -l go
python -m run.summarization.finetune.python_wan.eval -l php
python -m run.summarization.finetune.python_wan.eval -l ruby
python -m run.summarization.finetune.python_wan.eval -l java
python -m run.summarization.finetune.python_wan.eval -l javascript
python -m run.summarization.finetune.python_wan.eval -l csn