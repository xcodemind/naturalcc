# step 1
1) make directory(```~/.ncc/py150/seqrnn/raw```)
2) download [raw py150_file data](https://files.sri.inf.ethz.ch/data/py150_files.tar.gz)
3) unzip download data
```
mkdir -p ~/.ncc/py150/seqrnn/raw
cd ~/.ncc/py150/seqrnn/raw
wget https://files.sri.inf.ethz.ch/data/py150_files.tar.gz

# unzip py150 data
tar -zxvf py150_files.tar.gz
tar -zxvf data.tar.gz
```


# step 2
generate token (```train/test.tok```) and type (```train/test.ids```) sequence file of raw ast
```
# for train data
python -m dataset.py150.seqrnn.prepare -i ~/.ncc/py150/seqrnn/raw/python100k_train.txt -o ~/.ncc/py150/seqrnn/raw/train
# for eval data
python -m dataset.py150.seqrnn.prepare -i ~/.ncc/py150/seqrnn/raw/python50k_eval.txt  -o ~/.ncc/py150/seqrnn/raw/test
```

# step 3
run python to generate raw/bin(now unavailable) data and dictionary
```
python -m dataset.py150.seqrnn.preprocess
```