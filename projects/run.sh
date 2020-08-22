#!/usr/bin/env bash

# code search net
python -m dataset.csn.flatten
python -m dataset.csn.feature_extract
python -m dataset.csn.filter

# code-nn
 python -m dataset.codenn.csharp.__main__
python -m dataset.csn.feature_extract -l csharp -f ~/.ncc/codenn/flatten -r ~/.ncc/codenn/refine -s ~/.ncc/codenn/lib
python -m dataset.csn.filter -l csharp -r ~/.ncc/codenn/refine -f ~/.ncc/codenn/filter

