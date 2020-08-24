#!/usr/bin/env bash

# code search net
python -m dataset.csn_msra.flatten
python -m dataset.csn_msra.feature_extract
python -m dataset.csn_msra.filter

# code-nn
python -m dataset.codenn.csharp.__main__
python -m dataset.csn_msra.feature_extract -l csharp -f ~/.ncc/codenn/flatten -r ~/.ncc/codenn/refine -s ~/.ncc/codenn/lib
python -m dataset.csn_msra.filter -l csharp -r ~/.ncc/codenn/refine -f ~/.ncc/codenn/filter

# binarization dataset
python -m dataset.csn_msra.summarization.preprocess_multiattrs -l ruby
python -m dataset.csn_msra.summarization.preprocess_multiattrs -l go
python -m dataset.csn_msra.summarization.preprocess_multiattrs -l python
python -m dataset.csn_msra.summarization.preprocess_multiattrs -l php
python -m dataset.csn_msra.summarization.preprocess_multiattrs -l java
python -m dataset.csn_msra.summarization.preprocess_multiattrs -l javascript
python -m dataset.csn_msra.summarization.preprocess_multiattrs -l csharp
# create a shared dataset for multi-language
python -m dataset.csn_msra.summarization.preprocess_multilingual
