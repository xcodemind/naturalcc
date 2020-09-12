#!/usr/bin/env bash

#javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz:
#  https://drive.google.com/file/d/1YfHvacsAn9ngfjiJYbdo8LiFUqkbroxj/view
#javascript_augmented.pickle.gz:
#  https://drive.google.com/file/d/1YfPTPPOv4evldpN-n_4QBDWDWFImv7xO/view

data_names=(
  "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz"
  "javascript_augmented.pickle.gz"
)
data_urls=(
  "1YfHvacsAn9ngfjiJYbdo8LiFUqkbroxj"
  "1YfPTPPOv4evldpN-n_4QBDWDWFImv7xO"
)
for (( idx = 0 ; idx < ${#data_names[@]} ; idx++ )); do

echo "Downloading augmented_javascript dataset: ${data_names[idx]}"
DIR=~/.ncc/augmented_javascript/raw/
mkdir -p ${DIR}
FILE=${DIR}${data_names[idx]}

if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    fileid=${data_urls[idx]}
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    gunzip ${FILE} -d ${DIR} # && rm ${FILE}
fi

done


#"https://drive.google.com/file/d/1YtLVoMUsxU6HTpu5Qvs0xldm_SC_FZRz/view"
#"https://drive.google.com/file/d/1YvoM6rxcaX1wsyQu0HbGurQdaV6fsmym/view"
#"https://drive.google.com/file/d/1YsoSKGhuOUw3CNAAzZ3j9Fm5C8itc8Jt/view"
#"https://drive.google.com/file/d/1YunIabuWqd3V9kZssloXrOUvyfLcM7FH/view"
#"https://drive.google.com/file/d/1YvN5UQuijRgUAL3aF1MzGLn1NiVMTAEo/view"

data_names=(
  "1YtLVoMUsxU6HTpu5Qvs0xldm_SC_FZRz"
  "1YvoM6rxcaX1wsyQu0HbGurQdaV6fsmym"
  "1YsoSKGhuOUw3CNAAzZ3j9Fm5C8itc8Jt"
  "1YunIabuWqd3V9kZssloXrOUvyfLcM7FH"
  "1YvN5UQuijRgUAL3aF1MzGLn1NiVMTAEo"
)
data_names=(
  "target_wl"
  "test_nounk.txt"
  "test_projects_gold_filtered.json"
  "train_nounk.txt"
  "valid_nounk.txt"
)

for (( idx = 0 ; idx < ${#data_names[@]} ; idx++ )); do

echo "Downloading augmented_javascript dataset files: ${data_names[idx]}"
DIR=~/.ncc/augmented_javascript/type_prediction/raw/
mkdir -p ${DIR}
FILE=${DIR}${data_names[idx]}

if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    fileid=${data_urls[idx]}
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
fi

done