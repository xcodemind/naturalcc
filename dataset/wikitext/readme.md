## Data Preprocessing
### Step 0. Obtaining dataset

```
cd ~/.ncc
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
cd ~/.ncc/wikitext-103-raw
mkdir raw
mv *.raw raw/
```

### Step 1. BPE tokenization

```
cd ~/.ncc/wikitext-103-raw
mkdir gpt2_bpe
cd gpt2_bpe
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.tdxt
```

.raw => .bpe

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.train.raw --outputs ~/.ncc/wikitext-103-raw/train.bpe --keep-empty --workers 50
```

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.valid.raw --outputs ~/.ncc/wikitext-103-raw/valid.bpe --keep-empty --workers 50
```

```
python -m dataset.wikitext.multiprocessing_bpe_encoder --inputs ~/.ncc/wikitext-103-raw/raw/wiki.test.raw --outputs ~/.ncc/wikitext-103-raw/test.bpe --keep-empty --workers 50
```

### Step 2. Generate dataset for model training (raw or mmap mode)

.bpe => bin

1. (Suggested!) MMAP mode (default). Mainly set the dataset-impl and destdir

```
python -m dataset.wikitext.preprocess --only-source --srcdict ~/.ncc/wikitext-103-raw/gpt2_bpe/dict.txt --trainpref ~/.ncc/wikitext-103-raw/train.bpe --validpref ~/.ncc/wikitext-103-raw/valid.bpe --testpref ~/.ncc/wikitext-103-raw/test.bpe --destdir ~/.ncc/wikitext-103-raw/data-mmap --task masked_lm --dataset-impl mmap --workers 50 
```


2. Raw mode. Mainly set the dataset-impl and destdir
```
python -m dataset.wikitext.preprocess --only-source --srcdict ~/.ncc/wikitext-103-raw/gpt2_bpe/dict.txt --trainpref ~/.ncc/wikitext-103-raw/train.bpe --validpref ~/.ncc/wikitext-103-raw/valid.bpe --testpref ~/.ncc/wikitext-103-raw/test.bpe --destdir ~/.ncc/wikitext-103-raw/data-raw --task masked_lm --dataset-impl raw --workers 50
```

