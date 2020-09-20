
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
mv /export/share/jianguo/scodebert/augmented_javascript/raw/javascript_augmented.json /export/share/jianguo/scodebert/augmented_javascript/contracode/data-raw/no_augmented/train.code
```

## Step 4: Download ContraCode's SPM Vocab
```
cd /export/share/jianguo/scodebert/augmented_javascript/contracode/data-raw/no_augmented/
wget https://contrastive-code.s3.amazonaws.com/codesearchnet_javascript/csn_unigrams_8k_9995p.tar.gz
tar -xzf csn_unigrams_8k_9995p.tar.gz
# you will have csnjs_8k_9995p_unigram_url.vocab csnjs_8k_9995p_unigram_url.model
cut -f1 csnjs_8k_9995p_unigram_url.vocab | tail -n +10 | sed "s/$/ 100/g" > csnjs_8k_9995p_unigram_url.dict.txt
```


## Step 3(Deprecated): Run sentencepiece to obtain the vocabulary model

```
python -m dataset.augmented_javascript.run_sentencepiece
```

Move the generated `.model` and `.vocab` to the `no_augmented` and `augmented` folders

```
cp csnjs_8k_9995p_unigram_url.vocab csnjs_8k_9995p_unigram_url.model no_augmented/
mv csnjs_8k_9995p_unigram_url.vocab csnjs_8k_9995p_unigram_url.model augmented/
```


## Step 4(Deprecated): Sentencepiece-format vocabulary to Fairseq-format vocabulary

Change to the folder of `csnjs_8k_9995p_unigram_url.vocab` (`[default data directory]/augmented_javascript/contracode/data-raw/no_augmented` in my server), and tranform it into fairseq-format.

```
cd /export/share/jianguo/scodebert/augmented_javascript/contracode/data-raw/no_augmented
cut -f1 csnjs_8k_9995p_unigram_url.vocab | tail -n +10 | sed "s/$/ 100/g" > csnjs_8k_9995p_unigram_url.dict.txt
```



## Step 5: Preprocessing
> Note: currently only 100 samples are preprocessed for debugging. Modify around line 123 of ```preprocess.py```.

If we want to pretrain the codebert, we will use this data.
```
python -m dataset.augmented_javascript.preprocess
```

If we want to pretrain via contrastive learning, we should use this dataset with augmentation.
```
python -m dataset.augmented_javascript.preprocess_augmented
```

