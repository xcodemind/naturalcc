# step1 
download and precess dataset from [CodeSearchNet](https://github.com/github/CodeSearchNet)

* you can set your args at this file [download_and_process_dataset.py](./download_and_process_dataset.py)

```
python download_and_process_dataset.py
```

data demo (Ref: [CodeSearchNet](https://github.com/github/CodeSearchNet)):
+ repo: the owner/repo
+ path: the full path to the original file
+ func_name: method name
+ original_string: code snippet
+ language: the programming language
+ **code**: code snippet
+ **code_tokens**: the sequence modal of a code snippet
+ **docstring**: comment
+ **docstring_tokens**: tokenized version of docstring
+ sha: this field is not being used [TODO: add note on where this comes from?]
+ partition: {train, valid, test}
+ url: the url for the code snippet including the line numbers

*blod arttributes (code, code_tokens, docstring_tokens) are crucial for our program*

```
{
  'code': 
        'public void setServletRegistrationBeans(
                    Collection<? extends ServletRegistrationBean<?>> servletRegistrationBeans) {
                Assert.notNull(servletRegistrationBeans,
                        "ServletRegistrationBeans must not be null");
                this.servletRegistrationBeans = new LinkedHashSet<>(servletRegistrationBeans);
            }',
  'code_tokens': 
        ['publi   c', 'void', 'setServletRegistrationBeans', '(', 'Collection', '<', '?', 
        'extends', 'ServletRegistrationBean', '<', '?', '>', '>',
        'servletRegistrationBeans', ')', '{', 'Assert', '.', 'notNull', '(', 
        'servletRegistrationBeans', ',', '"ServletRegistrationBeans must not be null"',
        ')', ';', 'this', '.', 'servletRegistrationBeans', '=', 'new', 'LinkedHashSet',
        '<>', '(', 'servletRegistrationBeans', ')', ';', '}'],
  'docstring': 
        'Set {@link ServletRegistrationBean}s that the filter will be registered against.
        @param servletRegistrationBeans the Servlet registration beans',
  'docstring_tokens': 
        ['Set', '{'],
}
```

# step 2
build AST Parser

You can build your own AST Parser file
1) download parser programs from [tree-sitter](https://github.com/tree-sitter). <br>
*In this program, we have already download C/C++/C#/Go/Python/Java/Javascript/Ruby/PHP parser files.*
2) unzip those parser program files <br>
```unzip ./parser_zips/\*.zip -d your_dir/```
3) build your own AST Parser file, run ```python build_parser_file.py```


or, use our built AST Parser file [languages.so](parser_zips/), which support Java/Javascript/Go/Python/Ruby/PHP

# Step 3
parse code snippets, and generate code/comment dicts
1) parse comment: sometimes [docstring_tokens] are wrong, like: <br>
```
docstring = \
'Set {@link ServletRegistrationBean}s that the filter will be registered against.

@param servletRegistrationBeans the Servlet registration beans'
```

```
docstring_tokens = ['Set', '{']
```

In such situation, we choose to parse first line of docstring as our [docstring_tokens], like:

```Set {@link ServletRegistrationBean}s that the filter will be registered against.```

then, filter operators and parse it into

 ```['set', 'link', 'servlet', 'registration', 'beans', 'that', 'the', 'filter', 'will', 'be', 'registered', 'against']```

Filter rules:
1) if [docstring_tokens] ends with '(', '{', '[', parse our own--parse [docstring]:
     skip [docstring] containing non-ASCII code
     remove url
     only consider first sentence of [docstring]
     remove @link etc. or {}
2) skip [docstring_tokens] containing non-ASCII code
3) remove '-/*/~/='+ in [docstring_tokens/docstring]
4) '!/\`'+ -> '!/\`'
     
     

<br>
2) parse code: parse [code_tokens] as our sequence modal

Filter rules:
1) remove comment(like, '//', '/* */') in code
2) skip code if its tokens' length is too long


<br>
3) parse AST tree: parse [code] to generate AST tree

Filter rules:
1) if language is PHP, we should add '<?php ' at the head of [code]
2) build AST tree in dict format with tree_sitter
3) remove comment node in AST tree
4) remove nodes with single children node

 
# Step 4
run commands


argurments:
```
--language: programming language, [java/javascript/python/php/ruby/go]
--dataset_dir: dataset save direction
--dataset_name: dataset type, [raw/base/deepcom2/deepcom]
    * raw: raw ASTs
    * base: binary ASTs with leaf nodes padded
    * deepcom: ASTs with leaf nodes padded and SBTs
    * deepcom2: ASTs with leaf nodes padded and our SBTs
--core_num: multi-processing cpu core num

--debug: debug mode, [True/False]
--debug_data_size: dataset size of debug mode, we recommend small number
```

i. build raw dataset<br>
```
python -u ./dataset/parse_raw_dataset.py --language XXX --dataset_dir XXX --dataset_name XXX --core_num XXX (--debug XXX --debug_data_size XXX)
```


ii. build base dataset<br>
```
python -u ./dataset/parse_base_dataset.py --language XXX --dataset_dir XXX --dataset_name XXX --core_num XXX (--debug XXX --debug_data_size XXX)
```

iii. build deepcom2 dataset<br>
```
python -u ./dataset/parse_deepcom2_dataset.py --language XXX --dataset_dir XXX --dataset_name XXX --core_num XXX (--debug XXX --debug_data_size XXX)
```

iv. build deepcom dataset<br>
```
python -u ./dataset/parse_deepcom_dataset.py --language XXX --dataset_dir XXX --dataset_name XXX --core_num XXX (--debug XXX --debug_data_size XXX)
```


or build raw/base/deepcom2/deepcom dataset in shell script
```
sh ./dataset/run.sh
```

<br>

# Step 5
Dataset info

| languages | train(ours/all) | file num | valid(ours/all) | file num | test(ours/all) | file num | max sub-token len |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| go | 172202/317843 | 11 | 8917/14243 | 1 | 7723/14292 | 1 | 32 |
| java | 206604/454467 | 16 | 7353/1532 | 1 | 11932/26910 | 1 | 32 |
| javascript | 35933/123894 | 5 | 2480/8254 | 1 | 1910/6484 | 1 | 35 |
| php | 170201/523730 | 18 | 8028/26016 | 1 | 8512/28392 | 1 | 41 |
| python | 174764/412192 | 14 | 9594/23108 | 1 | 9442/22177 | 1 | 18 |
| ruby | 29815/48793 | 2 | 1322/2210 | 1 | 1288/2280 | 1 | 11 |


