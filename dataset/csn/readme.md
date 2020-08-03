## Raw Data to Model Input
> All the running commands here should be executed in the root of project folder (the path of your `naturalcodev3`).
For example, in my environment I will stay at `/data/wanyao/Dropbox/ghproj-titan/naturalcodev3`.

#### Step1: ```prepare.py``` to download dataset and Tree-Sitter libraries. <br>

```
python -m dataset.csn.prepare
```

#### Step 2: ```flatten.py``` to flatten important attributes of raw CodeSearchNet dataset. <br>

```
python -m dataset.csn.flatten
```

#### Step 3: ```preprocess.py``` to generate new attributes of flatten data. Need to update. <br>

```
python -m dataset.csn.preprocess
```