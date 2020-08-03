#!/usr/bin/env bash

mkdir -p ~/.ncc/codenn/raw
cd ~/.ncc/codenn/raw

wget https://github.com/sriniiyer/codenn/raw/master/data/stackoverflow/csharp/train.txt
wget https://github.com/sriniiyer/codenn/raw/master/data/stackoverflow/csharp/valid.txt
wget https://github.com/sriniiyer/codenn/raw/master/data/stackoverflow/csharp/test.txt
