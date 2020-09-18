# CodeSearchNet dataset

**we recommend to run this repository on linux/macOS**

### step 1. download CSN raw dataset (```~/.ncc/raw```) and parse files (```~/.ncc/libs```), and unzip them. <br>
```
python -m dataset.csn.download -l [language] -d [raw data directory] -b [library directory]
```

command to download and process all language files
```
python -m dataset.csn.download
```
after you run this command, your directories ```~/.ncc/code_search_net/raw``` and ```~/.ncc/code_search_net/libs``` would be
```
(base) yang@GS65:~/.ncc/code_search_net/raw$ ls
go  go.zip  java  javascript  javascript.zip  java.zip  php  php.zip  python  python.zip  ruby  ruby.zip

(base) yang@GS65:~/.ncc/code_search_net/libs$ ls
go.so           java.so   python.so   tree-sitter-c-sharp-master.zip  tree-sitter-java-master.zip        tree-sitter-php-master.zip     tree-sitter-ruby-master.zip
go.zip          java.zip  python.zip  tree-sitter-go-master           tree-sitter-javascript-master      tree-sitter-python-master
javascript.so   php.so    ruby.so     tree-sitter-go-master.zip       tree-sitter-javascript-master.zip  tree-sitter-python-master.zip
javascript.zip  php.zip   ruby.zip    tree-sitter-java-master         tree-sitter-php-master             tree-sitter-ruby-master
```


### step 2. flatten attributes of code snippets into different files. For instance, flatten ruby's code_tokens into ```train/valid/test.code_tokens```.
```
python -m dataset.csn.flatten -l [language] -d [raw data directory] -f [flatten data directory] -a [data attributes] -c [cpu cores]
```
command to flatten attributes of all language files
```
python -m dataset.csn.flatten
```
after you run this command, your directory ```~/.ncc/code_search_net/flatten``` would be

```
(base) yang@GS65:~/.ncc/code_search_net/flatten$ ls
go  java  javascript  php  python  ruby

(base) yang@GS65:~/.ncc/code_search_net/flatten$ ls ruby/
test              test.docstring         test.original_string  train.code_tokens       train.func_name        valid.code         valid.docstring_tokens
test.code         test.docstring_tokens  train                 train.docstring         train.original_string  valid.code_tokens  valid.func_name
test.code_tokens  test.func_name         train.code            train.docstring_tokens  valid                  valid.docstring    valid.original_string
```
*If you only use code/code_tokens/docstring/docstring_tokens*

### step 3(optional). If you want to get AST/binary-AST etc. of code and so on. Plz run such command.
```
python -m dataset.csn.feature_extract -l [language] -f [flatten data directory] -r [refine data directory] -s [parse file] -a [data attributes] -c [cpu cores]
```
-*Attention: default attributes is ```[code, code_tokens, docstring, docstring_tokens]```. If you want to generate AST and others plz set ```-a code, code_tokens, docstring, docstring_tokens, xxx ```. Some attributes depend on others, for instance, path depends on ast and ast depend on raw_ast; therefore, you have to set ```-a code, code_tokens, docstring, docstring_tokens raw_ast path ```. Not all attributes can be generated, if it failed, we replace this line with None and will be filtered at [step 4] *-
```
a mapping to generate new attributes of code snippet.
Examples:
"raw_ast" <= "code",    # raw_ast, an AST contains all info of a code, e.g. comment, single root node, ...
"ast" <= "raw_ast",     # ast, saving leaf nodes into "value" nodes and non-leaf nodes into "children" nodes
"path" <= "ast",        # path, a path from a leaf node to another leaf node 
"sbt" <= "raw_ast",     # sbt, a depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
"sbtao" <= "sbt'",       # sbtao, an improved depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
"binary_ast" <= "raw_ast", # bin_ast, an sophisticated binary AST, remove nodes with single child, tokenize leaf node and padding with <PAD>(for DGL Lib.)
"traversal" <= "ast",   #DFS traversal
```

command to extract new attributes of a language's new attributes of code/docstring etc..
```
python -m dataset.csn.feature_extract
```
after you run this command, your directory ```~/.ncc/code_search_net/refine``` would be
```
(base) yang@GS65:~/.ncc/code_search_net/refine$ ls
ruby
(base) yang@GS65:~/.ncc/code_search_net/refine$ ls ruby/
test.code         test.docstring         train.code         train.docstring         train.raw_ast  valid.code_tokens  valid.docstring_tokens
test.code_tokens  test.docstring_tokens  train.code_tokens  train.docstring_tokens  valid.code     valid.docstring
```

 
### step 4. filter data containing invalid attributes. Only available after you run step3.
```
python -m dataset.csn.filter -l [language] -r [refined data directory] -f [filter data directory] -a [data attributes]
```

command to filter those code snippets whose attributes are invalid. For instance, if a train code snippet CANNNOT be exracted into raw_ast, in train\.raw_ast, such code is written as None but its other attributes are valid. While filtering, its attributes will be dropped becuase such code snippet contains a invalid raw_ast line.
```
python -m dataset.csn.filter
```
after you run this command, your directory ```~/.ncc/code_search_net/refine``` would be
```
(base) yang@GS65:~/.ncc/code_search_net/filter$ ls
ruby
(base) yang@GS65:~/.ncc/code_search_net/filter$ ls ruby/
test.code         test.docstring         train.code         train.docstring         valid.code         valid.docstring
test.code_tokens  test.docstring_tokens  train.code_tokens  train.docstring_tokens  valid.code_tokens  valid.docstring_tokens
```
### Note

To process all the languages, one just needs to run the commands before `-l`, e.g., using `python -m dataset.csn.download` to download all datasets.

### Example of only using code/code_tokens/docstring/docstring_tokens
```
python -m dataset.csn.download
python -m dataset.csn.flatten
```

### Example of only using code/code_tokens/docstring/docstring_tokens/ast
```
python -m dataset.csn.download
python -m dataset.csn.flatten
python -m dataset.csn.feature_extract -a code code_tokens docstring docstring_tokens raw_ast ast
python -m dataset.csn.filter -a code code_tokens docstring docstring_tokens raw_ast ast
```