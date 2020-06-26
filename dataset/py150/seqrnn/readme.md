# step 1
download [raw py150_file data](https://files.sri.inf.ethz.ch/data/py150_files.tar.gz), <br>
and unzip tar.gz file at ```~/.ncc/py150/seqrnn/raw``` and its ```data.tar.gz```<br>
rename train/test data: ```mv python100k_train.txt train.txt```, ```mv python50k_eval.txt test.txt```<br>

# step 2
run python to generate token (```train/test.tok```) and type (```train/test.type```) sequence file of raw ast
```
python -m dataset.py150.seqrnn.token_type -i ~/.ncc/py150/seqrnn/raw/train.txt # for train data
python -m dataset.py150.seqrnn.token_type -i ~/.ncc/py150/seqrnn/raw/test.txt # for test data
```

# step 3
run python to generate raw/bin data and dictionary
```
python -m dataset.py150.seqrnn.preprocess
```