# CodeSearchNet dataset

**we recommend to run this repository on linux/macOS**

### step 1. download csn_msra raw dataset (```~/.ncc/raw```)
```
bash dataset/csn_msra/download.sh
```

### step 2. download parse files (```~/.ncc/libs```), and build them. 
```
python -m dataset.csn_msra.build -l [language] -b [library directory]
```

### step 3. flatten attributes of code snippets into different files. For instance, flatten ruby's code_tokens into 
```train/valid/test.code_tokens```.
```
python -m dataset.csn_msra.flatten -l [language] -d [raw data directory] -f [flatten data directory] -a [data attributes] -c [cpu cores]
```

### step 3(optional). extract features of data attributes. For instance, AST, binary-AST etc. of code.
```
python -m dataset.csn_msra.feature_extract -l [language] -f [flatten data directory] -r [refine data directory] -s [parse file] -a [data attributes] -c [cpu cores]
```
 
### step 4. filter data containing invalid attributes.
```
python -m dataset.csn_msra.filter -l [language] -r [refined data directory] -f [filter data directory] -a [data attributes]
```

### step 5. finalize dataset. change config/*.yml config 
generate data-raw dataset
```
dataset_impl: raw
destdir: ~/.ncc/csn_icse21/100k/summarization/data-raw/go
```
generate data-mmap dataset
```
dataset_impl: mmap
destdir: ~/.ncc/csn_icse21/100k/summarization/data-mmap/go
```


