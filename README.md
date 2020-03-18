# naturalcodev2
## Note
This copy of code is private now, please do not distribute it. Thanks.

We are planning to release part of this copy of code in the next year, after we submit a demo paper to ICSE2021.

naturalcode version 2.0
dataset from: https://github.com/github/CodeSearchNet

* [dataset](dataset): processed dataset file
* [demo](demo): demo display
* [doc](doc): some description about this program
* [eval](eval): evaluation codes
* [exp](exp): codes for draw graphs
* [run](run): run scripts

建立自己的run脚本文件夹。必须要有一个main.py定义你的训练过程，一个yaml文件定义你的参数。生成的test文件也务必放到这个目录下。

【例子】[baseline](run/summarization/unil/sl)
* [script](script): run scripts
* [src](src): source code

[src/data](src/data)dict字典，code/comment的token_dicts

[src/dataset](src/dataset)base某个数据集的某个mode下的数据集，可以组合为某个数据集的train/valid/test的合集UniDataloader，
进一步组成TLDataloader适用于迁移学习

[src/log](src/log)定义log的输出，不建议改变啊

[src/metric](src/metric)定义loss，最好一个文件，一个大类别的loss


[src/model](src/model)定义模型，之后会进一步改进。暂时没用

[src/module](src/module)定义encoder和decoder之类

[src/opt](src/opt)即将删除

[src/trainer](src/trainer)定义一个模型的train/eval的pipeline，和train过程

[src/utils](src/utils)只要是高频引用函数/变量就放进来，新建也没关系。日后，添加一个readme.md，将util的所有信息保存进来


----------------------------------------------
naturalcode v1.0: https://github.com/wanyao1992/naturalcode <br>
This program handle our old dataset--including C, Python, Java, C#--crawled from github.com


# Attention
一些经验之谈<br>
* 打算使用yaml作为输入参数。一般而言一个模型有多个不同的参数输入，建议建立一个文件夹，main.py作为run脚本的主体，*.yaml作为不同的参数。
* BaseDataset是单一语言的某一个数据集（例如train）的；UniDataset是单一语言的train/valid/test的数据集；TLDataset是迁移学习数据集涉及source/target数据集
* run/debug/sl.yml是测试样例。source有效，target=None时，就是baseline数据集；source有效，target!=None时，会自动将一些source的信息内容复制到target中


100-small dataset
```
dataset_dir: /data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100_small

data num:
{
"python":{"train":20052,"valid":3256,"test":2047},
"java":{"train":20416,"valid":2870,"test":2137},
"go":{"train":19585,"valid":3046,"test":1913},
"php":{"train":18818,"valid":9117,"test":1927},
"ruby":{"train":14879,"valid":1475,"test":1466},
"javascript":{"train":17234,"valid":2750,"test":1833}
}

avg tok len:
{
"python":{"train":28.5981946938,"valid":33.3667076167,"test":30.9892525647},
"java":{"train":31.4527331505,"valid":37.8027874564,"test":33.3827795976},
"go":{"train":26.674853204,"valid":32.6336178595,"test":30.6084683743},
"php":{"train":37.5207248379,"valid":104.4468575189,"test":39.8495070057},
"ruby":{"train":38.2428254587,"valid":62.6922033898,"test":64.2864938608},
"javascript":{"train":42.5135778113,"valid":54.912,"test":52.9530823786}
}

avg ast node len:
{
"python":{"train":54.3910831837,"valid":62.1572481572,"test":57.7738153395},
"java":{"train":54.6490007837,"valid":63.8850174216,"test":57.7140851661},
"go":{"train":46.0949195813,"valid":56.9409061064,"test":52.1939362258},
"php":{"train":67.0785949623,"valid":78.1332675222,"test":70.3783082512},
"ruby":{"train":62.1433564084,"valid":73.5708474576,"test":70.9208731241},
"javascript":{"train":72.1616571893,"valid":84.6836363636,"test":85.537370431}
}                   

avg path len:
{
"python":{"train":6.8566756433,"valid":6.9738267813,"test":6.923234001},
"java":{"train":7.7438195533,"valid":7.7470940767,"test":7.7552643893},
"go":{"train":7.4930436559,"valid":7.2961457649,"test":7.3356194459},
"php":{"train":7.4137007121,"valid":7.5030426675,"test":7.4553399066},
"ruby":{"train":7.013475368,"valid":7.042820339,"test":6.9850886767},
"javascript":{"train":7.1381838227,"valid":7.24168,"test":7.2244735406}
}                                            

avg comment len:
{
"python":{"train":10.3488430082,"valid":10.7392506143,"test":10.6507083537},
"java":{"train":11.9257934953,"valid":13.4972125436,"test":11.0477304633},
"go":{"train":11.1101353076,"valid":12.8066316481,"test":12.5969681129},
"php":{"train":8.6070251886,"valid":9.0923549413,"test":8.874416191},
"ruby":{"train":11.6025942604,"valid":12.253559322,"test":12.2012278308},
"javascript":{"train":11.2059301381,"valid":11.4123636364,"test":12.0212765957}
}                       
```


Dataset Info

```
dataset_dir:/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100

data num:
{
"python":{"train":233644,"valid":12790,"test":12829},
"java":{"train":241146,"valid":8363,"test":14698},
"go":{"train":208237,"valid":10051,"test":9419},
"php":{"train":194776,"valid":9117,"test":10019},
"ruby":{"train":33674,"valid":1475,"test":1466},
"javascript":{"train":59800,"valid":4054,"test":3045}
}

avg tok len:
{
"python":{"train":96.3818630053,"valid":104.3054730258,"test":96.7328708395},
"java":{"train":94.5832897912,"valid":75.2301805572,"test":95.0432031569},
"go":{"train":75.7721202284,"valid":63.2522137101,"test":78.1431149804},
"php":{"train":96.4721936994,"valid":104.4468575189,"test":97.902485278},
"ruby":{"train":62.8285027024,"valid":62.6922033898,"test":64.2864938608},
"javascript":{"train":129.8413210702,"valid":122.0064134188,"test":118.3159277504}
}

avg ast node len:
{
"python":{"train":71.2433017754,"valid":71.1463643471,"test":71.6318497155},
"java":{"train":78.0698124787,"valid":77.3031208896,"test":77.7296911144},
"go":{"train":70.2054341928,"valid":76.0160183066,"test":73.3977067629},
"php":{"train":77.8022189592,"valid":78.1332675222,"test":75.7309112686},
"ruby":{"train":71.1106491655,"valid":73.5708474576,"test":70.9208731241},
"javascript":{"train":73.4977926421,"valid":76.8537247163,"test":75.508045977}
}

avg path len:
{
"python":{"train":7.0563509442,"valid":7.0511368256,"test":7.0647049653},
"java":{"train":7.8719488609,"valid":7.7819466699,"test":7.8545897401},
"go":{"train":7.6392740963,"valid":7.4768500647,"test":7.5218600701},
"php":{"train":7.4873868444,"valid":7.5030426675,"test":7.4840143727},
"ruby":{"train":7.044074954,"valid":7.042820339,"test":6.9850886767},
"javascript":{"train":7.1007655518,"valid":7.1529748397,"test":7.1299376026}
}

avg comment len:
{"python":{"train":11.4256561264,"valid":11.6766223612,"test":11.2631537922},
"java":{"train":12.5053038408,"valid":13.2372354418,"test":12.5021091305},
"go":{"train":13.9581870657,"valid":14.6172520147,"test":13.0107230067},
"php":{"train":9.5893128517,"valid":9.0923549413,"test":9.4124164088},
"ruby":{"train":12.0449308072,"valid":12.253559322,"test":12.2012278308},
"javascript":{"train":11.768729097,"valid":11.7257030094,"test":12.2791461412}
}
```