## Dataset
```
cd dataset/codesearchnet
```

0. Explore the dataset if you are interested

```
python explore_dataset.py
```

1. Build a paser for a specific programming language, e.g., Ruby

```
python build_parser_file.py
```

After running this command, you will obtain a ruby.so file

2. Prepare the dataset. In this step, you will flatten the dataset from the raw json files, and extract the ASTs offline.
```
python prepare_summarization.py
```

3. Preprocess the dataset for training. In this step, we will obtain the dataset which can be fed into the model for training.

```
python preprocess.py
```


## Training
Go back to the ROOT folder of project, and then run the following commands.

```
cd run/summarization/seq2seq
```

1. Config the ruby.yml according to your requirement.

2. Start to training.

```
python main.py
```

## Testing
