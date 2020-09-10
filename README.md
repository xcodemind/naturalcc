# NaturalCode-V3.0
Naturalcode is a sequence modeling toolkit that allows researchers and developers to train custom models for many software engineering tasks, e.g., code summarization, code retrieval and code clone detection. Our vision is to bridge the gap between programming language and natural language via some machine learning techniques.

## Note
This copy of code is private now, please do not distribute it. Thanks.

<!-- We are planning to release part of this copy of code in the next year, after we submit a demo paper to ICSE2021. -->

## What's New:
Sep. 2020: support CodeBert.

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

<!-- To install naturalcode: -->

<!-- ``` -->
<!-- pip install naturalcode -->
<!-- ``` -->
#### Step 1: Clone it
```
git clone https://github.com/whatsmyname/naturalcodev3
```

#### Step 2: To install naturalcode from source and develop locally

```
cd naturalcodev3
pip install --editable .
pip install -r requirements.txt
```

## Dataset
Currently, we have processed the following datasets:

- codesearchnet (see `dataset/csn`, and the `dataset/codesearchnet` is deprecated)
- py150 (see `dataset/py150`)
- codenn_charp (see `dataset/codenn_csharp`)
- wikitext (see `dataset/wikitext` for Roberta pretraining.)
- iwslt14 (see `dataset/iwstl14` for machine translation)


## Runing
> All the running commands here should be executed in the root of project folder (the path of your `naturalcodev3`).
For example, in my environment I will stay at `/data/wanyao/Dropbox/ghproj-titan/naturalcodev3`.



### CodeBert and SCodeBert

#### Step 1: Download the raw dataset and process it into data that can be fed to models.
For our current SCodeBert task, we mainly use the CodeSearchNet dataset to fairly compare with CodeBert.

Please refer to [dataset/csn/readme.md](https://github.com/whatsmyname/naturalcodev3/tree/master/dataset/csn) to process the CodeSearchNet dataset.
After this step, we will obtain the preprocessed data in `~/.ncc/` folder.

One quick step to download all datasets: 

```python -m dataset.csn.download 
```

or add `-l language` to download the dataset for a specific language. 

#### Step 2: Pre-training

##### Model 1 ([code-roberta](https://github.com/whatsmyname/naturalcodev3/tree/master/run/codebert/code_roberta)): only code tokens, Roberta architecture

```
python -m run.codebert.code_roberta.train
```
##### Model 2 (code-docstring-roberta): code tokens and comment tokens, Roberta architecture

```
python -m run.codebert.code_docstring_roberta.train
```

##### Model 3 (traverse-roberta): only code structure (AST traverse), Roberta architecture

```
python -m run.codebert.traverse_roberta.train
```
##### Model 4 (traverse-docstring-roberta): code structure (AST traverse) and comment tokens, Roberta architecture

```
python -m run.codebert.traverse_docstring_roberta.train
```

#### Step 3: Fine-tuning
##### On code summarization
##### Model 1 (traverse-roberta): only code structure (AST traverse), Roberta architecture

```
python -m run.codebert.traverse_roberta.ft_summarization
```

##### On code retrieval
> TODO

#### Step 4: Evaluation

##### On code summarization
```
python -m run.codebert.traverse_roberta.summarize
```

##### On code retrieval
> TODO

### Code Completion
TODO

### Code Summarization
TODO

### Code Retrieval
TODO

<!-- ## Organization -->
<!-- * [dataset](dataset): processed dataset file -->
<!-- * [demo](demo): demo display -->
<!-- * [doc](doc): some description about this program -->
<!-- * [eval](eval): evaluation codes -->
<!-- * [exp](exp): codes for draw graphs -->
<!-- * [run](run): run scripts -->


## License
naturalcode is MIT-licensed. The license applies to the pre-trained models as well.

## Citation
Please cite as:
xxx