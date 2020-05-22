## Data Preprocessing
### Step 1. BPE tokenization

1. Raw mode. Mainly set the outputs

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.train.raw --outputs ~/.ncc/wikitext-103-raw/data-raw/train.bpe --keep-empty --workers 10
```

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.valid.raw --outputs ~/.ncc/wikitext-103-raw/data-raw/valid.bpe --keep-empty --workers 10
```

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.test.raw --outputs ~/.ncc/wikitext-103-raw/data-raw/test.bpe --keep-empty --workers 10
```

2. (Suggested!) MMAP mode (default). Mainly set the outputs


```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.train.raw --outputs ~/.ncc/wikitext-103-raw/data-mmap/train.bpe --keep-empty --workers 10
```

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.valid.raw --outputs ~/.ncc/wikitext-103-raw/data-mmap/valid.bpe --keep-empty --workers 10
```

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.test.raw --outputs ~/.ncc/wikitext-103-raw/data-mmap/test.bpe --keep-empty --workers 10
```

### Step 2. Generate dataset for model training (raw or mmap mode)

1. Raw mode. Mainly set the dataset-impl and destdir
```
python -m dataset.wikitext.preprocess --only-source --srcdict ~/.ncc/wikitext-103-raw/gpt2_bpe/dict.txt --trainpref ~/.ncc/wikitext-103-raw/wiki.train.bpe --validpref ~/.ncc/wikitext-103-raw/wiki.valid.bpe --testpref ~/.ncc/wikitext-103-raw/wiki.test.bpe --destdir ~/.ncc/wikitext-103-raw/data-raw --task masked_lm --dataset-impl raw --workers 10 > ~/.ncc/wikitext-103-raw/log/log.preprocess
```

2. (Suggested!) MMAP mode (default). Mainly set the dataset-impl and destdir

```
python -m dataset.wikitext.preprocess --only-source --srcdict ~/.ncc/wikitext-103-raw/gpt2_bpe/dict.txt --trainpref ~/.ncc/wikitext-103-raw/wiki.train.bpe --validpref ~/.ncc/wikitext-103-raw/wiki.valid.bpe --testpref ~/.ncc/wikitext-103-raw/wiki.test.bpe --destdir ~/.ncc/wikitext-103-raw/data-mmap --task masked_lm --dataset-impl mmap --workers 10 > ~/.ncc/wikitext-103-raw/log/log.preprocess
```