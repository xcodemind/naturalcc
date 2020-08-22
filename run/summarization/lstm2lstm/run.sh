#!/usr/bin/env bash

languages=(ruby go python php java javascript csharp)
for lang in $languages
do
  echo "running" $lang
  #  nohup python -m run.summarization.lstm2lstm.train -l $lang > run/summarization/lstm2lstm/$lang.log 2>&1
done