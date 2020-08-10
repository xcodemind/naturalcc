#Command:
```
# generate ruby path dataset
python -m dataset.csn.preprocess -l ruby -a raw_ast ast path -c 20

# generate ruby path/docstring_tokens mmap dataset
python -m dataset.csn.base.preprocess

run code2seq model
nohup python -m run.summarization.code2seq.train > run/summarization/code2seq/ruby.log 2>&1 &
```

 
