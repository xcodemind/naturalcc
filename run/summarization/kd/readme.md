## Teacher training for each programming language

Make ensure the following parameters are set correctly.
- criterion: label_smoothed_cross_entropy
- is_distill: False

```
python -m run.summarization.kd.train
```

## Distillation
Make ensure the following parameters are set correctly.
- criterion: distill_label_smoothed_cross_entropy
- is_distill: True

```
python -m run.summarization.kd.train
```


## Evaluation
