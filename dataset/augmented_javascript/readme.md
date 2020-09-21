
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
```

(Deprecated)If we want to pretrain via contrastive learning, we should use this dataset with augmentation.
```
python -m dataset.augmented_javascript.preprocess_augmented
```