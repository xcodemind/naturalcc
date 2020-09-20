python -m dataset.csn.retrieval.csn_tokenization # split code_tokens and joint docstring_tokens
python -m dataset.csn.retrieval.bpe_docstring_tokens # bpe joint docstring_tokens
python -m dataset.csn.retrieval.individual # binarize split.code_tokens/bpe.docstring_tokens