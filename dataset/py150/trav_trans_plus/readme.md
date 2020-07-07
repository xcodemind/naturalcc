# step 1
generate data (```~/.ncc/py150/trav_trans/raw```) from raw data (```~/.ncc/py150/raw```)
```
python -m dataset.py150.generate_new_trees -i ~/.ncc/py150/raw/python100k_train.json -o ~/.ncc/py150/raw/train.json # for train data
python -m dataset.py150.generate_new_trees -i ~/.ncc/py150/raw/python50k_eval.json -o ~/.ncc/py150/raw/test.json # for test data
```

Note:
if you have already generated data for ```trav_trans```, you can copy raw data from its directory
```
makdir -p ~/.ncc/py150/trav_trans_plus/
cp ~/.ncc/py150/trav_trans/raw ~/.ncc/py150/trav_trans_plus/
```

# step 2
copy data from  ```~/.ncc/py150/raw``` to ```~/.ncc/py150/trav_trans/raw```
```
mkdir -p ~/.ncc/py150/trav_trans/raw/
cp ~/.ncc/py150/raw/train.json ~/.ncc/py150/trav_trans/raw/train.ast_trav_df
cp ~/.ncc/py150/raw/test.json ~/.ncc/py150/trav_trans/raw/test.ast_trav_df
```

# step 3
generate AST DFS tokens and ids
```
python -m dataset.py150.trav_trans.preprocess
```
