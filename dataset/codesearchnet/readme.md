# CodeSearchNet(CSN) Dataset Generation

## 1) Download Tree-Sitter Libraries
1) Download Tree-Sitter libraries from ```https://codeload.github.com/tree-sitter/tree-sitter-*/zip/master``` into ```~/.ncc/CodeSearchNet/libs/``` 
2) Extract Tree-Sitter libraries at the download direction
3) Build AST parser files with Tree-Sitter libraries and save them at ```~/.ncc/CodeSearchNet/so/*.so```

## 2) Download CSN Raw Dataset
1) Download CSN Raw Dataset from ```https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/*.zip``` into ```~/.ncc/CodeSearchNet/raw/```
2) Extract CSN Raw Dataset *.jsonl.gz files from raw direction into ```~/.ncc/CodeSearchNet/data/```

## 3) Flatten CSN Raw Data and parse Original AST
1) Flatten *.jsonl.gz into flatten direction into ```~/.ncc/CodeSearchNet/flatten/```
2) Parse code into AST with AST parser files into ```~/.ncc/CodeSearchNet/flatten/```

## 4) Merge Data of Same Parition[train/valid/test] and attributes into one
1) Use cat command to merge those files into ```~/.ncc/CodeSearchNet/flatten/``` <br>
e.g. ```~/.ncc/CodeSearchNet/flatten/ruby/train/index/*.txt``` => ```~/.ncc/CodeSearchNet/flatten/ruby/train/train.index```

# Command
```
cd this_project
python -m dataset.codesearch.codesearch
```
how to use? Please, follow the instruction of ```dataset/codesearch/codesearch.py/__main__```