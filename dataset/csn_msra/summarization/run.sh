#!/usr/bin/env bash

# finalize dataset on each language's dictionaries
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs_100k -l ruby > log  2>&1 &
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs_100k -l go > go.log  2>&1 &
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs_100k -l python > python.log  2>&1 &
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs_100k -l php > php.log  2>&1 &
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs_100k -l java > java.log  2>&1 &
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs_100k -l javascript > javascript.log  2>&1 &
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs_100k -l csharp > csharp.log  2>&1 &

# finalize dataset on a shared dictionary of CSN and C# dataset
nohup python -m dataset.csn_msra.summarization.preprocess_multilingual_100k > all.log  2>&1 &
