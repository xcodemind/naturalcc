# Dataet: java_hu

The authors of [A Transformer-based Approach for Source Code Summarizatio
n](https://arxiv.org/pdf/2005.00653.pdf) shared their code and dataset at https://github.com/wasiahmad/NeuralCodeSum.

In this repo., its offers original and runnable codes of Java dataset and therefore we can generate AST with Tree-Sitter.

However, as for Python dataset, its original codes are not runnable. An optional way to deal with such problem is that
  we can acquire runnable Python codes from raw data(https://github.com/wanyao1992/code_summarization_public).

# step 1 
```
bash dataset/java_hu/download.sh
```

# step 2
```
python -m dataset.java_hu.flatten
```

# step 3
```
python -m dataset.csn_msra.feature_extract -l java -f ~/.ncc/java_hu/flatten -r ~/.ncc/java_hu/refine -s ~/.ncc/java_hu/libs 
```

# step 4
```
python -m dataset.csn_msra.filter -l java -r ~/.ncc/java_hu/refine -f ~/.ncc/java_hu/filter
```