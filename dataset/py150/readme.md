# Steps to reproduce

## Step 1
Download dataset **Version 1.0 [526.6MB]** from (https://www.sri.inf.ethz.ch/py150)


## Step 2
to install requirements run
```
pip install -r requirements.txt
```

## Step 3
to generate new trees according to the paper run
```
python generate_new_trees.py -i /home/wanyao/.ncc/py150/python50k_eval.json -o /home/wanyao/.ncc/py150/new_python50k_eval.json
```

## Step 4
to generate vocabulary according to the paper run
```
python generate_vocab.py -i /home/wanyao/.ncc/py150/new_python100k_train.json -o /home/wanyao/.ncc/py150/new_python100k_train.pkl -t ast
```

python generate_vocab.py -i /data/wanyao/ghproj_d/code-prediction-transformer/py150/new_python1k_train.json -o /data/wanyao/ghproj_d/code-prediction-transformer/py150/data-bin/new_python1k_train.pkl -t ast

## Step 5
to generate data according to the README.md
```
python -m models.trav_trans_plus.generate_data -a /home/wanyao/.ncc/py150/new_python100k_train.json -o /home/wanyao/.ncc/py150/new_new_python100k_train.txt
```
python -m models.trav_trans.generate_data -a /data/wanyao/ghproj_d/code-prediction-transformer/py150/new_python1k_train.json -o /data/wanyao/ghproj_d/code-prediction-transformer/py150/data-bin/new_new_python1k_train.txt

# Step 6
to generate ast ids according to the README.md
```
python -m models.trav_trans.generate_ast_ids -a /home/wanyao/.ncc/py150/new_python100k_train.json -o /home/wanyao/.ncc/py150/generated_ids.txt all
```

python -m models.trav_trans.generate_ast_ids -a /data/wanyao/ghproj_d/code-prediction-transformer/py150/new_python1k_train.json -o /data/wanyao/ghproj_d/code-prediction-transformer/py150/data-bin/generated_ids.txt all
