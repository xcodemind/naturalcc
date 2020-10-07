#!/usr/bin/env bash


python -m dataset.csn_feng.flatten

# tokenization
python -m projects.kd.tokenization.subtoken

python -m projects.kd.tokenization.share
python -m projects.kd.tokenization.individual -f javascript
python -m projects.kd.tokenization.individual -f ruby

python -m projects.kd.tokenization.individual -f python_wan.subtoken
python -m projects.kd.tokenization.individual -f ruby.subtoken


cd ~/.ncc/kd/code_tokens_docstring_tokens/data-mmap

cp *.json go/
cp *.json ruby/
cp *.json java/
cp *.json javascript/
cp *.json python/
cp *.json php/
cp *.json java_hu/
cp *.json python_wan/

# BPE
python -m projects.kd.bpe.json2txt
python -m projects.kd.bpe.build_bpe
python -m projects.kd.bpe.share

cd ~/.ncc/kd/code_docstring/

cp *.json data-mmap/go/
cp *.json data-mmap/ruby/
cp *.json data-mmap/java/
cp *.json data-mmap/javascript/
cp *.json data-mmap/python/
cp *.json data-mmap/php/
cp *.json data-mmap/java_hu/
cp *.json data-mmap/python_wan/
