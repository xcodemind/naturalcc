# Preprocess for CodeBert

### Step 1: get special symbols
```
python -m dataset.csn.codebert.get_special_symbols
```

### Step 2: generate BPE dictionary
```
# BPE core config
# --language ruby
# --vocab-size 50000
# --modalities ['code', ]

python -m dataset.csn.codebert.run_sentencepiece
```

### Step 3: generate mmap data
```
python -m dataset.csn.codebert.preprocess_codebert.py # from data-raw to data-mmap
```