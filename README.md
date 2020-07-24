# NaturalCode-V3.0
Naturalcode is a sequence modeling toolkit that allows researchers and developers to train custom models for many software engineering tasks, e.g., code summarization, code retrieval and code clone detection. Our vision is to bridge the gap between programming language and natural language via some machine learning techniques.

## Note
This copy of code is private now, please do not distribute it. Thanks.

We are planning to release part of this copy of code in the next year, after we submit a demo paper to ICSE2021.

## What's New:
May 2020: support CodeBert.

## Features:
- CNN
- LSTM
- Transformer
- Reinforcement Learning (e.g., policy gradient, self-critical training and actor-critic network)
- GAN
- AREL

## Requirements and Installation
- PyTorch version >= 1.4.0
- Python version >= 3.6
- For training new models, you'll also need an NVIDIA GPU and NCCL
- For faster training install NVIDIA's apex library with the --cuda_ext and --deprecated_fused_adam options

To install naturalcode:

```
pip install naturalcode
```

To install naturalcode from source and develop locally:

```
git clone https://github.com/whatsmyname/naturalcodev3
cd naturalcodev3
pip install --editable .
```

## Dataset
Currently, we have processed the following datasets:

- codesearchnet

Please refer to `dataset/codesearchnet/readme.md` to process the `codesearchnet` dataset.


- py150
- wikitext
- iwslt14

## Runing

Please refer to the corresponding readme file in the `ncc/run` folder.

## Organization
* [dataset](dataset): processed dataset file
* [demo](demo): demo display
* [doc](doc): some description about this program
* [eval](eval): evaluation codes
* [exp](exp): codes for draw graphs
* [run](run): run scripts


## License
naturalcode is MIT-licensed. The license applies to the pre-trained models as well.

## Citation
Please cite as:
xxx