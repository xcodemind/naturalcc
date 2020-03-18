#!/usr/bin/env bash

### 和 run_script_lowercase.sh 区别在于把那个文件里 train 替换成fse20ruby100_v2_train , valid 替换成fse20ruby100_v2_valid ， test替换成 fse20ruby100_v2_test

cd boost_1_64_0/
./bootstrap.sh
./b2 -j4 --prefix=$PWD --libdir=$PWD/lib64 --layout=system link=static install || echo FAILURE

cp giza-pp/GIZA++-v2/GIZA++ giza-pp/GIZA++-v2/snt2cooc.out \
   giza-pp/mkcls-v2/mkcls installmoses/tools

/data/wanyao/work/baseline/installmoses

/data/wanyao/work/baseline


# 上面是一些安装命令，不用管
#本文件参照http://www.statmt.org/moses/?n=Moses.Baseline




cd /data/wanyao/work/baseline/corpus
/data/wanyao/work/baseline/installmoses/scripts/tokenizer/escape-special-chars.perl < training/moses_data_lower.fse20ruby100_v2_train.com \
    > moses_data_lower.fse20ruby100_v2_train.esc.com
/data/wanyao/work/baseline/installmoses/scripts/tokenizer/escape-special-chars.perl < training/moses_data_lower.fse20ruby100_v2_train.code \
    > moses_data_lower.fse20ruby100_v2_train.esc.code

/data/wanyao/work/baseline/installmoses/scripts/recaser/train-truecaser.perl \
    --model /data/wanyao/work/baseline/corpus/truecase-model.com --corpus     \
    /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.esc.com
/data/wanyao/work/baseline/installmoses/scripts/recaser/train-truecaser.perl \
    --model /data/wanyao/work/baseline/corpus/truecase-model.code --corpus     \
    /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.esc.code

/data/wanyao/work/baseline/installmoses/scripts/recaser/truecase.perl \
   --model /data/wanyao/work/baseline/corpus/truecase-model.com         \
   < /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.esc.com \
   > /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.code-com.true.com
/data/wanyao/work/baseline/installmoses/scripts/recaser/truecase.perl \
   --model /data/wanyao/work/baseline/corpus/truecase-model.code         \
   < /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.esc.code \
   > /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.code-com.true.code


/data/wanyao/work/baseline/installmoses/scripts/training/clean-corpus-n.perl \
    /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.code-com.true code com \
    /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.code-com.clean 1 100


cd /data/wanyao/work/baseline/lm
/data/wanyao/work/baseline/installmoses/bin/lmplz -o 3 </data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.code-com.true.com >\
     moses_data_lower.fse20ruby100_v2_train.code-com.arpa.com

/data/wanyao/work/baseline/installmoses/bin/build_binary \
   moses_data_lower.fse20ruby100_v2_train.code-com.arpa.com \
   moses_data_lower.fse20ruby100_v2_train.code-com.blm.com

echo "is this an English sentence ?"                       \
   | /data/wanyao/work/baseline/installmoses/bin/query moses_data_lower.fse20ruby100_v2_train.code-com.blm.com


cd /data/wanyao/work/baseline/workinglow
nohup nice /data/wanyao/work/baseline/installmoses/scripts/training/train-model.perl -root-dir train \
-corpus /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_train.code-com.clean                             \
-f code -e com -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:3:/data/wanyao/work/baseline/lm/moses_data_lower.fse20ruby100_v2_train.code-com.blm.com:8                          \
-external-bin-dir /data/wanyao/work/baseline/installmoses/tools -cores 30 >& training.out &


########################################    tuning


cd /data/wanyao/work/baseline/corpus
/data/wanyao/work/baseline/installmoses/scripts/tokenizer/escape-special-chars.perl < training/moses_data_lower.fse20ruby100_v2_valid.com \
    > moses_data_lower.fse20ruby100_v2_valid.esc.com
/data/wanyao/work/baseline/installmoses/scripts/tokenizer/escape-special-chars.perl < training/moses_data_lower.fse20ruby100_v2_valid.code \
    > moses_data_lower.fse20ruby100_v2_valid.esc.code

/data/wanyao/work/baseline/installmoses/scripts/recaser/truecase.perl --model truecase-model.com \
   < moses_data_lower.fse20ruby100_v2_valid.esc.com > moses_data_lower.fse20ruby100_v2_valid.true.com
/data/wanyao/work/baseline/installmoses/scripts/recaser/truecase.perl --model truecase-model.code \
   < moses_data_lower.fse20ruby100_v2_valid.esc.code > moses_data_lower.fse20ruby100_v2_valid.true.code

cd /data/wanyao/work/baseline/workinglow
nohup nice /data/wanyao/work/baseline/installmoses/scripts/training/mert-moses.pl \
/data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_valid.true.code /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_valid.true.com \
/data/wanyao/work/baseline/installmoses/bin/moses train/model/moses.ini --mertdir /data/wanyao/work/baseline/installmoses/bin/ \
--decoder-flags="-threads 38" &> mert.out &


#############################    Testing



mkdir /data/wanyao/work/baseline/workinglow/binarised-model
cd /data/wanyao/work/baseline/workinglow

/data/wanyao/work/baseline/installmoses/bin/processPhraseTableMin \
-in train/model/phrase-table.gz -nscores 4 \
-out binarised-model/phrase-table
/data/wanyao/work/baseline/installmoses/bin/processLexicalTableMin \
-in train/model/reordering-table.wbe-msd-bidirectional-fe.gz \
-out binarised-model/reordering-table



# cp -f  注意要用 -f参数，因为目标路径可能有同名的之前生成出来的文件

cp -f /data/wanyao/work/baseline/workinglow/mert-work/moses.ini  /data/wanyao/work/baseline/workinglow/binarised-model/


将 /data/wanyao/work/baseline/workinglow/binarised-model/moses.ini  中的
   PhraseDictionaryMemory 改成 PhraseDictionaryCompact
    PhraseDictionary 设置为

         /data/wanyao/work/baseline/workinglow/binarised-model/phrase-table.minphr

    LexicalReordering 设置为

          /data/wanyao/work/baseline/workinglow/binarised-model/reordering-table.minlexr




cd /data/wanyao/work/baseline/corpus
/data/wanyao/work/baseline/installmoses/scripts/tokenizer/escape-special-chars.perl < training/moses_data_lower.fse20ruby100_v2_test.com \
    > moses_data_lower.fse20ruby100_v2_test.esc.com
/data/wanyao/work/baseline/installmoses/scripts/tokenizer/escape-special-chars.perl < training/moses_data_lower.fse20ruby100_v2_test.code \
    > moses_data_lower.fse20ruby100_v2_test.esc.code

/data/wanyao/work/baseline/installmoses/scripts/recaser/truecase.perl --model truecase-model.com \
< moses_data_lower.fse20ruby100_v2_test.esc.com > moses_data_lower.fse20ruby100_v2_test.true.com
/data/wanyao/work/baseline/installmoses/scripts/recaser/truecase.perl --model truecase-model.code \
< moses_data_lower.fse20ruby100_v2_test.esc.code > moses_data_lower.fse20ruby100_v2_test.true.code


cd /data/wanyao/work/baseline/workinglow
/data/wanyao/work/baseline/installmoses/scripts/training/filter-model-given-input.pl             \
filtered-moses_data_lower.fse20ruby100_v2_test binarised-model/moses.ini /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_test.true.code \
-Binarizer /data/wanyao/work/baseline/installmoses/bin/processPhraseTableMin

nohup nice /data/wanyao/work/baseline/installmoses/bin/moses            \
-f /data/wanyao/work/baseline/workinglow/filtered-moses_data_lower.fse20ruby100_v2_test/moses.ini   \
< /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_test.true.code                \
> /data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v2_test.translated.com         \
2> /data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v2_test.out
/data/wanyao/work/baseline/installmoses/scripts/generic/multi-bleu.perl \
-lc /data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_test.true.com              \
< /data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v2_test.translated.com

