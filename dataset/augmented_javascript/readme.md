
## Step 0: Download augmented_javascript dataset realeased in https://github.com/parasj/contracode. 
```
bash download.sh 
```

## Step 1: Cast the downloaded target_dict to support our scenario
```
python -m dataset.augmented_javascript.cast_target_dict 
```

## Step 1: Cast the downloaded type_prediction_data to support our scenario
```
python -m dataset.augmented_javascript.cast_type_prediction_data
```

## Step 1: Cast downloaded `.pkl` file to `.json` for data binarization (mmap) (~1min).
```
python -m dataset.augmented_javascript.cast_pkl2json
```

## Step 2: Run sentencepiece to obtain the vocabulary and corresponding model (~20min).
Run the sentencepiece
```
python -m dataset.augmented_javascript.run_sentencepiece
```
Cast the sentencepiece vocab to the format of NCC Dictionary.
- For Yao & Yang
```
cd ~/.ncc/augmented_javascript/codebert/code_roberta/data-mmap 
cut -f1 csnjs_8k_9995p_unigram_url.vocab | tail -n +10 | sed "s/$/ 100/g" > csnjs_8k_9995p_unigram_url.dict.txt
```
- For Jian-Guo
```
xxxx
```

## Step 3: Preprocessing (< 1 h).
> Note: currently only 100 samples are preprocessed for debugging. Modify around line 123 of ```preprocess.py```.

If we want to pretrain the codebert, we will use this data. Check the paths in `preprocess.yml` before running this command.
```
python -m dataset.augmented_javascript.preprocess
```

If we want to obtain the AST and its traverse features (i.e., for `traverse_roberta`), continue.

Copy `javascript/train.code` to `~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/`
```
mkdir -p ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/
cd ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/
cp ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/train.code ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/
```

Build `javascript.so`
```
python -m dataset.augmented_javascript.build
```

Extract features
```
python -m dataset.csn.feature_extract -l javascript -f ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented -r ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/refine -s ~/.ncc/augmented_javascript/libs -a code raw_ast ast traversal -c 40
```

Filter out those cannot generate AST
```
python -m dataset.csn.filter -l javascript -r ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/refine -f ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter -a code ast traversal
```
(Not for current) If we want to pretrain via contrastive learning, we should use this dataset with augmentation.
```
python -m dataset.augmented_javascript.preprocess_augmented
```