# CodeSearchNet dataset

**we recommend to run this repository on linux/macOS**

### step 1. download CSN raw dataset (```~/.ncc/raw```) and parse files (```~/.ncc/libs```), and unzip them. 
```
python -m dataset.csn.download -l [language] -d [raw data directory] -b [library directory]
```

### step 2. flatten attributes of code snippets into different files. For instance, flatten ruby's code_tokens into 
```train/valid/test.code_tokens```.
```
python -m dataset.csn.flatten -l [language] -d [raw data directory] -f [flatten data directory] -a [data attributes] -c [cpu cores]
```

### step 3(optional). extract features of data attributes. For instance, AST, binary-AST etc. of code.
```
python -m dataset.csn.feature_extract -l [language] -f [flatten data directory] -r [refine data directory] -s [parse file] -a [data attributes] -c [cpu cores]
```
 
### step 4. filter data containing invalid attributes.
```
python -m dataset.csn.filter -l [language] -r [refined data directory] -f [filter data directory] -a [data attributes]
```

### Note

To process all the languages, one just needs to run the commands before `-l`, e.g., using `python -m dataset.csn.download` to download all datasets.
