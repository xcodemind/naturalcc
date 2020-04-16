## Dataset
```
cd dataset/translation
```


1. Prepare the dataset.
```
chmod +x prepare-iwslt14.sh
./prepare-iwslt14.sh
```

2. Preprocess the dataset for training. In this step, we will obtain the dataset which can be fed into the model for training.

```
python preprocess.py
```


## Training
Go back to the ROOT folder of project, and then run the following commands.

```
cd run/translation/seq2seq
```

1. Config the iwlt14.de-en.yml according to your requirement.

2. Start to training.

```
python main.py
```

## Testing
