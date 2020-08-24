#!/usr/bin/env bash

# download refined dataset from CodeSearchNet
mkdir -p ~/.ncc/CodeSearchNet_icse21/raw
cd ~/.ncc/CodeSearchNet_icse21/raw
# download
pip install gdown
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
#rm Cleaned_CodeSearchNet.zip
