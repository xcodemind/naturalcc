#Command:
```
# run before you flatten CodeSearchNet dataset and generate path raw dataset. 
python -m dataset.csn.base.preprocess
```

# Path data generation
A code snippet contains many paths. If those paths are less than we expected (e.g. in this repo, we save 300 paths of
 each code snippet), copy some paths; if more than, we sample some of those paths. <br>

For fast load data, we flatten a path into 3 list and save them into 3 lines of a json file so that we can binarize
 them for fast load and write. Given we save 300 paths of a code snippet, 900 lines presents path data of a code.  
 

  
Paths of a code snippet should be following:
```
# Example: [var_size](terminals), [var_size, ..., return](path), [return](terminals)
[var, size], [var_size, ..., return], [return]

# while saving
[var, size], 
[var_size, ..., return], 
[return],
```
 
 
