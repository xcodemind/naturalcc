# Seq2Seq for code summarization task
This 

running with float32
```shell script
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/python_wan > run/summarization/seq2seq/config/seq2seq.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.train -f config/python_wan
```
running with float16
```shell script
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/python_wan.fp16 > run/summarization/seq2seq/config/seq2seq.fp16.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.train -f config/python_wan.fp16
```