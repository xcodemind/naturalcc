#!/usr/bin/env bash

# train teacher
nohup python -m run.summarization.kd.seq2seq.train_teacher -l ruby > run/summarization/kd/seq2seq/teacher.ruby.log 2>&1
nohup python -m run.summarization.kd.seq2seq.train_teacher -l python > run/summarization/kd/seq2seq/teacher.python.log 2>&1
nohup python -m run.summarization.kd.seq2seq.train_teacher -l php > run/summarization/kd/seq2seq/teacher.php.log 2>&1
nohup python -m run.summarization.kd.seq2seq.train_teacher -l go > run/summarization/kd/seq2seq/teacher.go.log 2>&1
nohup python -m run.summarization.kd.seq2seq.train_teacher -l javascript > run/summarization/kd/seq2seq/teacher.javascript.log 2>&1
nohup python -m run.summarization.kd.seq2seq.train_teacher -l java > run/summarization/kd/seq2seq/teacher.java.log 2>&1

# train student: to
nohup python -m run.summarization.kd.seq2seq.train_student -l csn > run/summarization/kd/seq2seq/student.csn.log 2>&1


# finetune
nohup python -m run.summarization.kd.seq2seq.finetune > run/summarization/kd/seq2seq/kd.csn.csharp.log 2>&1


# eval
nohup python -m run.summarization.kd.seq2seq.eval > run/summarization/kd/seq2seq/kd.eval.csn2csharp.log 2>&1