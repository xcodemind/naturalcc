

## Step 0: Download augmented_javascript dataset realeased in https://github.com/parasj/contracode. 
```
bash download.sh 
```
## Step 0: cast the downloaded target_dict to support our scenario
```
python -m dataset.augmented_javascript.cast_target_dict 
```

## Step 0: cast the downloaded type_prediction_data to support our scenario
```
python -m dataset.augmented_javascript.cast_type_prediction_data
```


## Step 1: Run sentencepiece to obtain the vocabulary model

```
python -m dataset.augmented_javascript.run_sentencepiece
```


## Step 2: Sentencepiece-format vocabulary to Fairseq-format vocabulary

Change to the folder of `csnjs_8k_9995p_unigram_url.vocab` (`~/.ncc/augmented_javascript/contracode/data-raw/no_augmented` in my server), and tranform it into fairseq-format.

```
cd ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented
cut -f1 csnjs_8k_9995p_unigram_url.vocab | tail -n +9 | sed "s/$/ 100/g" > csnjs_8k_9995p_unigram_url.dict.txt
```



## Step 3: Preprocessing
> Note: currently only 100 samples are preprocessed for debugging. Modify around line 116 of ```preprocess.py```.

If we want to pretrain the codebert, we will use this data.
```
python -m dataset.augmented_javascript.preprocess
```

If we want to pretrain via contrastive learning, we should use this dataset with augmentation.
```
python -m dataset.augmented_javascript.preprocess_augmented
```

