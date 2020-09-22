# Preprocess for CodeBert

### step1, generate BPE dictionary
```
# BPE core config
# --language ruby
# --vocab-size 50000
# --modalities ['code', ]

python -m dataset.codesearchnet.codebert.preprocess_wordpiece
```

### step2, generate mmap data
```
python -m dataset.codesearchnet.codebert.preprocess_codebert.py # from data-raw to data-mmap
```


# Preprocess for HiCodeBert

The flatten data is in ~/.ncc/code_search_net/flatten/ruby/
1. Insert <S_SEP> token to the flatten code based on the args\['preprocess'\]\['inserted'\]
e.g., train.code -> train_inserted.code

```
<CLS> w1 w2 wn <S_SEP> w11 w22 wnn <S_SEP> ...
```

2. build vocabulary from the train_inserted.code

3. build dataset based on the vocabulary.

