# how to run
1) ```prepare.py``` to download dataset and Tree-Sitter libraries. <br>
```python -m dataset.csn.prepare```
2) ```flatten.py``` to flatten important attributes of raw CodeSearchNet dataset. <br>
```python -m dataset.csn.flatten```
3) ```preprocess.py``` to generate new attributes of flatten data. Need to update. <br>
```python -m dataset.csn.preprocess```