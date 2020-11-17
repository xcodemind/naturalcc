# NaturalCC
NaturalCC is a sequence modeling toolkit that allows researchers and developers to train custom models for many software engineering tasks, e.g., code summarization, code retrieval and code clone detection. 
Our vision is to bridge the gap between programming language and natural language via some machine learning techniques.

<p align="center">
    <img src="https://img.shields.io/badge/version-0.4.0-green" alt="Version">
</p>
<hr>

## Note
This copy of code is private now, please do not distribute it. Thanks.

<!-- We are planning to release part of this copy of code in the next year, after we submit a demo paper to ICSE2021. -->


## Features
- [Code Summarization](run/summarization)
- [Code Retrieval](run/retrieval)
- [Type Inference](run/type_prediction)
- [Code Prediction](run/)

## Dataset
Currently, we have processed the following datasets:

- [Python_wan](dataset/python_wan/README.md)

*NCC supports Raw/Binarinzed dataset generation/loading*

## TBC:





## Requirements 
- PyTorch version >= 1.4.0
- Python version >= 3.6
- For training new models, you'll also need an NVIDIA GPU and NCCL
- For faster training install NVIDIA's apex library with the --cuda_ext and --deprecated_fused_adam options

## Installation
#### 1) Install [apex](https://github.com/NVIDIA/apex)
 to support half precision training.

```shell script
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

#### 2) Install other prerequisites libraries
```shell script
git clone https://github.com/xcodemind/naturalcc
cd naturalcc
pip install -r requirements.txt

# or install with conda 
# conda install --yes --file requirements.txt
```

#### 3) Install NCC
```shell script
python setup.py build_ext --inplace
```


## License
naturalcc is MIT-licensed. The license applies to the pre-trained models as well.

## Citation
Please cite as:
xxx