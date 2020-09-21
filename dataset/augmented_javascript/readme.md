
## Step 0: Download augmented_javascript dataset realeased in https://github.com/parasj/contracode. 
```
bash download.sh 
```

## Step 1: cast the downloaded target_dict to support our scenario
```
python -m dataset.augmented_javascript.cast_target_dict 
```

## Step 2: cast the downloaded type_prediction_data to support our scenario
```
python -m dataset.augmented_javascript.cast_type_prediction_data
```


## Step 3: Write pkl file to json file so that we can binarize dataset with multi-processing
```
python dataset/augmented_javascript/pkl2json.py
mkdir /export/share/jianguo/scodebert/augmented_javascript/contracode/data-raw/no_augmented/
# mv ~/.ncc/augmented_javascript/raw/javascript_augmented.json ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/train.code
mv /export/share/jianguo/scodebert/augmented_javascript/raw/javascript_augmented.json /export/share/jianguo/scodebert/augmented_javascript/contracode/data-raw/no_augmented/train.code
```

## Step 4: Run sentencepiece to obtain the vocabulary model

```
cp /export/share/jianguo/scodebert/augmented_javascript/raw/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz /export/share/jianguo/scodebert/augmented_javascript/contracode/data-raw/no_augmented/
# cp ~/.ncc/augmented_javascript/raw/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/
python -m dataset.augmented_javascript.run_sentencepiece
cut -f1 csnjs_8k_9995p_unigram_url.vocab | tail -n +10 | sed "s/$/ 100/g" > csnjs_8k_9995p_unigram_url.dict.txt
```


## Step 5: Preprocessing
> Note: currently only 100 samples are preprocessed for debugging. Modify around line 123 of ```preprocess.py```.

If we want to pretrain the codebert, we will use this data.
```
python -m dataset.augmented_javascript.preprocess

# AST
# copy javascript/train.code to ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/
mkdir -p ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/
cd ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/
cp ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/train.code ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/javascript/
# build javascript.so
python -m dataset.augmented_javascript.build
python -m dataset.csn.feature_extract -l javascript -f ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented -r ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/refine -s ~/.ncc/augmented_javascript/libs -a code raw_ast ast traversal -c 40
python -m dataset.csn.filter -l javascript -r ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/refine -f ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter -a code ast traversal

```

(Deprecated)If we want to pretrain via contrastive learning, we should use this dataset with augmentation.
```
python -m dataset.augmented_javascript.preprocess_augmented
```
