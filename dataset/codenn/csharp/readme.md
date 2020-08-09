```
# pip install antlr4
pip install antlr4-python3-runtime==4.5.2
# flatten code/docstring and generate code_tokens/docstring_tokens
python -m dataset.codenn.csharp.__main__
# generate new modality data
python -m dataset.csn.preprocess -l csharp -f ~/.ncc/codenn/csharp/flatten -s ~/.ncc/codenn/csharp/lib -a 'raw_ast ast path sbt sbtao binary_ast'
```