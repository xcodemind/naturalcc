
from __future__ import division
import torch
import os
import os.path

### dj
path_root = "/data/wanyao/work/ds/codedata/debug-data-path/sixshouldok/"
name_train_ct = "processed_all.train_ct.pt"
name_val_ct = "processed_all.val_ct.pt"
name_test_ct = "processed_all.test_ct.pt"

path_train_ct = path_root+name_train_ct
path_val_ct = path_root+name_val_ct
path_test_ct = path_root+name_test_ct

output_path = "/data/wanyao/work/baseline/corpus/training"


def write_one_line(line,fd):
    len_line = len(line)
    for s_idx in range(len_line):
        if s_idx != len_line - 1:
            fd.write(line[s_idx]+" ")
        else:
            fd.write(line[s_idx]+"\n")

def save_data_step_two(dict_data,f_code,f_comment):
    code_list = dict_data["code"]
    comment_list = dict_data["comment"]
    assert len(code_list) == len(comment_list)
    for index in range(len(code_list)):
        this_code_line = code_list[index]
        this_comment_line = comment_list[index]
        if index in [0,1,12999]:
            print("index: ",index)
            print("this_code_line: \n",this_code_line)
            print("this_comment_line: \n",this_comment_line)
        write_one_line(this_code_line, f_code)
        write_one_line(this_comment_line, f_comment)

def save_data_step_one(dict_data,dataset_type):
    output_code_name = "githubpythonformoses."+dataset_type+".code"
    output_com_name =  "githubpythonformoses."+dataset_type+".com"

    output_code_path = os.path.join(output_path, output_code_name)
    output_com_path = os.path.join(output_path, output_com_name)

    f_code = open(output_code_path, "w", encoding="utf-8")
    f_comment = open(output_com_path, "w", encoding="utf-8")

    save_data_step_two(dict_data, f_code, f_comment)
    f_code.close()
    f_comment.close()

print("output_path: \n",output_path)
print("path_train_ct: \n",path_train_ct)
print("path_val_ct: \n",path_val_ct)
print("path_test_ct: \n",path_test_ct)
train_ct, val_ct, test_ct = torch.load(path_train_ct), torch.load(path_val_ct), torch.load(path_test_ct)

print('save_data_step_one(train_ct,"train_ct") begin')
save_data_step_one(train_ct,"train_ct")
print('save_data_step_one(val_ct,"val_ct") begin')
save_data_step_one(val_ct,"val_ct")
print('save_data_step_one(test_ct,"test_ct") begin')
save_data_step_one(test_ct,"test_ct")

