# Step 0: convert traversal json file (a line of list) to txt file (a line of string), replace '\n' with [SEP]
```
python -m dataset.augmented_javascript.traversal.json2str -i ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/train.traversal
```

# Step 1: get AST non-lead node types and save them at ```.ast.node_types```
```
python -m dataset.augmented_javascript.traversal.get_type_node -i ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/train.ast -o ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/.ast.node_types
```

# Step 2: load AST non-lead node types and run sentencepiece at txt file
```
python -m dataset.augmented_javascript.traversal.run_sentencepiece -i ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/train.traversal.str -m ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/traversal.str -t ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript/.ast.node_types --vocab_size 8000
``` 

# Step 3: build dict from sentencepiece
```
cd ~/.ncc/augmented_javascript/contracode/data-raw/no_augmented/filter/javascript
cut -f1 traversal.str.vocab | tail -n +10 | sed "s/$/ 100/g" > traversal.str.dict.txt
```

# Step 4: binarize traversal dataset
```
python -m dataset.augmented_javascript.preprocess -f preprocess.traversal
```