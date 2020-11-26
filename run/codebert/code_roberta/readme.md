## MLM Pretraining

```
python -m run.codebert.code_roberta.train
```

## Task 1: Type Inference/Prediction
### Fine-tuning
```
python -m run.codebert.code_roberta.ft_type_predict
```

### Test downstream task on type prediction task
```
python -m run.codebert.code_roberta.type_predict
```


## Task 2: Code Summarization
### Fine-tuning
```
python -m run.codebert.code_roberta.ft_summarization
```

### Test downstream task on type prediction task
```
python -m run.codebert.code_roberta.summarize
```


## Task 3: Code Retrieval (TODO)
### Fine-tuning
```
python -m run.codebert.code_roberta.ft_retrieval
```

### Test downstream task on type prediction task
```
python -m run.codebert.code_roberta.retrieve
```