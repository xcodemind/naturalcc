#!/usr/bin/env bash


# finalize dataset on each language's dictionaries
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs -l ruby > ruby.log  2>&1
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs -l go > go.log  2>&1
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs -l python > python.log  2>&1
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs -l php > php.log  2>&1
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs -l java > java.log  2>&1
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs -l javascript > javascript.log  2>&1
nohup python -m dataset.csn_msra.summarization.preprocess_multiattrs -l csharp > csharp.log  2>&1

# finalize dataset on a shared dictionary of CSN and C# dataset
nohup python -m dataset.csn_msra.summarization.preprocess_multilingual > all.log  2>&1
