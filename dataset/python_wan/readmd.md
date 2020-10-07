# Dataet: Python_wan

The authors of [A Transformer-based Approach for Source Code Summarizatio
n](https://arxiv.org/pdf/2005.00653.pdf) shared their code and dataset at https://github.com/wasiahmad/NeuralCodeSum.

In this repo., its offers original and runnable codes of Java dataset and therefore we can generate AST with Tree-Sitter.

However, as for Python dataset, its original codes are not runnable. An optional way to deal with such problem is that
  we can acquire runnable Python codes from raw data(https://github.com/wanyao1992/code_summarization_public).

# step 1 
Download pre-processed and raw (python_wan) dataset.
```
bash dataset/python_wan/download.sh
```
# step 2
Replace ununnable original codes/docstrings with these from raw (python_wan) dataset.
```
python -m dataset.python_wan.src_code
```

# step 3
Move code/code_tokens/docstring/docstring_tokens to ```~/.ncc/python_wan/flatten/*```.
*you can download our flatten data [here](https://drive.google.com/drive/folders/1BCeLXVmUM5NTwLzdqf3zuiMQFhVPeg5S?usp=sharing)*
```
python -m dataset.python_wan.flatten
```

# step 4
```
python -m dataset.csn_feng.feature_extract -l python -f ~/.ncc/python_wan/flatten -r ~/.ncc/python_wan/refine -s ~/.ncc/python_wan/libs 
```

# step 5
```
python -m dataset.csn_feng.filter -l python -r ~/.ncc/python_wan/refine -f ~/.ncc/python_wan/filter
```
