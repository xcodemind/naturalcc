# Seq2Seq for code summarization task
This 

running with float32
```python
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.train -f config/python_wan
```
running with float16
```python
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.train -f config/python_wan.fp16
```