#!/usr/bin/env bash

# train teacher
nohup python -m run.summarization.kd.python_wan.train_teacher -l ruby > run/summarization/kd/python_wan/teacher.ruby.log 2>&1
nohup python -m run.summarization.kd.python_wan.train_teacher -l php > run/summarization/kd/python_wan/teacher.php.log 2>&1
nohup python -m run.summarization.kd.python_wan.train_teacher -l go > run/summarization/kd/python_wan/teacher.go.log 2>&1
nohup python -m run.summarization.kd.python_wan.train_teacher -l javascript > run/summarization/kd/python_wan/teacher.javascript.log 2>&1
nohup python -m run.summarization.kd.python_wan.train_teacher -l java > run/summarization/kd/python_wan/teacher.java.log 2>&1

nohup python -m run.summarization.kd.python_wan.train_teacher -l csn > run/summarization/kd/python_wan/teacher.csn.log 2>&1


# teacher generate topk indices/probilities and valid bleu
python -m run.summarization.kd.python_wan.teacher_generate -l ruby
python -m run.summarization.kd.python_wan.teacher_generate -l php
python -m run.summarization.kd.python_wan.teacher_generate -l go
python -m run.summarization.kd.python_wan.teacher_generate -l javascript
python -m run.summarization.kd.python_wan.teacher_generate -l java

# distill
nohup python -m run.summarization.kd.python_wan.train_student -l csn > run/summarization/kd/python_wan/student.csn2python.log 2>&1

# finetune
nohup python -m run.summarization.kd.python_wan.finetune -l python > run/summarization/kd/python_wan/kd_finetune.csn.python_wan.log 2>&1

# eval
nohup python -m run.summarization.kd.python_wan.eval -l python > run/summarization/kd/python_wan/kd.eval.csn2python.log 2>&1