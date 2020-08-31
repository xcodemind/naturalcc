#!/usr/bin/env bash


echo "Downloading Code Search Net(feng) dataset"
DIR=~/.ncc/code_search_net_feng/raw/
mkdir -p ${DIR}
FILE=${DIR}code_search_net_feng.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    # https://drive.google.com/open?id=13o4MiELiQoomlly2TCTpbtGee_HdQZxl
    fileid="1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} -d ${DIR} # && rm ${FILE}
fi

cd ${DIR}
raw_dataset=${DIR}CodeSearchNet
mv ${raw_dataset}/* ./
rm -fr ${raw_dataset}