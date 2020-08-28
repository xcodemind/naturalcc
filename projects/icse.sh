#!/usr/bin/env bash

# code search net
#bash dataset/csn_msra/download.sh
python -m dataset.csn_msra.build
python -m dataset.csn_msra.flatten
python -m dataset.csn_msra.feature_extractv2
python -m dataset.csn_msra.filter

# Python_wan
bash dataset/python_wan/download.sh
python -m dataset.python_wan.build
python -m dataset.python_wan.src_code
python -m dataset.python_wan.flatten
python -m dataset.csn_msra.feature_extract -l python -f ~/.ncc/python_wan/flatten -r ~/.ncc/python_wan/refine -s ~/.ncc/python_wan/libs
python -m dataset.csn_msra.filter -l python -r ~/.ncc/python_wan/refine -f ~/.ncc/python_wan/filter
python -m dataset.python_wan.portion

#binarize for debug and fast load
#baseline
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 1.0
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.8
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.7
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.6
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.5
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.4
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.3
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.2
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.1
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.05
python -m dataset.python_wan.summarization.preprocess_multiattrs -p 0.01

#csn => Python_wan, shared dict
python -m dataset.python_wan.summarization.preprocess_multilingual

## Java_hu
#bash dataset/java_hu/download.sh
#python -m dataset.java_hu.build
#python -m dataset.java_hu.src_code
#python -m dataset.java_hu.flatten
#python -m dataset.csn_msra.feature_extract -l java -f ~/.ncc/java_hu/flatten -r ~/.ncc/java_hu/refine -s ~/.ncc/java_hu/libs
#python -m dataset.csn_msra.filter -l java -r ~/.ncc/java_hu/refine -f ~/.ncc/java_hu/filter
#python -m dataset.java_hu.portion
#
##binarize for debug and fast load
##baseline
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 1.0
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.8
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.7
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.6
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.5
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.4
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.3
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.2
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.1
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.05
#python -m dataset.java_hu.summarization.preprocess_multiattrs -p 0.01
#
##csn => Java_hu, shared dict
#python -m dataset.java_hu.summarization.preprocess_multilingual