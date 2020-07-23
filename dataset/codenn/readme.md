# step1 download c# data at ```~/.ncc/codenn/raw```
```
./dataset/codenn/download.sh
```

# step2 flatten codeNN data into code/docstring file ```~/.ncc/codenn/flatten/*.code(docstring)```
```
python -m dataset.codenn.__main__
```