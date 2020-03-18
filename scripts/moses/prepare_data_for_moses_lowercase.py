
from __future__ import division
import torch
import os
import os.path
import glob
import ujson



# def save_data_step_two(dict_data,f_code,f_comment):
#     code_list = dict_data["code"]
#     comment_list = dict_data["comment"]
#     assert len(code_list) == len(comment_list)
#     for index in range(len(code_list)):
#         this_code_line = code_list[index]
#         this_comment_line = comment_list[index]
#         if index in [0,1,12999]:
#             print("print index in [0,1,12999]")
#             print("index: ",index)
#             print("this_code_line: \n",this_code_line)
#             print("this_comment_line: \n",this_comment_line)
#         write_one_line(this_code_line, f_code)
#         write_one_line(this_comment_line, f_comment)


def write_one_line(line,fd):
    len_line = len(line)
    for s_idx in range(len_line):
        # if line[s_idx] == '\n':
        #     assert False,print("line:====\n {}======\n",line)
        if '\n' in line[s_idx]:
            assert False,print("line:====\n {}======\n",line)
        if s_idx != len_line - 1:
            fd.write(line[s_idx].lower()+" ")
        else:
            fd.write(line[s_idx].lower()+"\n")

def load_file(filename: str):
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def load_data(dataset_files ) :
    data = {key: [] for key in dataset_files.keys()}
    for key, filenames in dataset_files.items():
        for fl in filenames:
            print("load_data , key:{} fl:{}".format(key,fl ))
            data[key].extend(load_file(fl))
    return data


# def read_input(path_root,load_keys,mode):
#
#     dataset_files=   {
#     key: sorted([filename for filename in
#                  glob.glob('{}/*'.format(os.path.join(path_root, key)))
#                  if mode in filename])
#     for key in load_keys}
#     print("dataset_files: ",dataset_files )
#     data = {key: [] for key in dataset_files.keys()}
#     for key, filenames in dataset_files.items():
#         for fl in filenames:
#             data[key].extend(load_file(fl))
#
#
#     return data


def read_input(path_root,load_keys,mode):
    print("read_input , mode: ",mode )
    dataset_files = {
        key: sorted([filename for filename in
                     glob.glob('{}/*'.format(os.path.join(path_root, key)))
                     if mode in filename])
        for key in load_keys
    }

    data = load_data(dataset_files)
    size = len(data['comment'])
    for key, value in data.items():
        print("key:{} len(value):{}".format(key,len(value)))
        assert size == len(value), Exception('{} data: {}, but others: {}'.format(key, len(value), size))

    return data

def save_data (dict_data,f_code,f_comment):
    code_list = dict_data["tok"]
    comment_list = dict_data["comment"]
    print(" len(code_list) :{}  len(comment_list): {}".format(len(code_list) , len(comment_list)))
    assert len(code_list) == len(comment_list)
    max_len_code = 0
    max_len_com = 0
    for index in range(len(code_list)):
        this_code_line = code_list[index]
        this_comment_line = comment_list[index]
        if len(this_code_line) > max_len_code:
            max_len_code = len(this_code_line)
        if len(this_comment_line) > max_len_com:
            max_len_com = len(this_comment_line)
        # print("save_data , index:{} \n this_code_line:{}\n this_comment_line:{}".format(index,this_code_line,this_comment_line))
        # print("write code line")
        write_one_line(this_code_line, f_code)
        # print("write comment line")
        write_one_line(this_comment_line, f_comment)
    print("max_len_code: {}  max_len_com:{} ".format(max_len_code,max_len_com))

def save_data_step_one(dict_data,output_code_path ,output_com_path ):


    f_code = open(output_code_path, "w", encoding="utf-8")
    f_comment = open(output_com_path, "w", encoding="utf-8")

    save_data(dict_data, f_code, f_comment)
    f_code.close()
    f_comment.close()

def clean(data):
    new = {}
    for k,v in data.items():
        n_v  =[]
        for i in range(len(v)):
            n_v.append([e.replace('\n',ENTER_TAG) for e in v[i]])
        new[k] = n_v
    return new



if __name__ == "__main__":
    #  python -m  scripts.moses.prepare_data_for_moses_lowercase
    #  python -m  scripts.moses.prepare_data_for_moses_lowercase | tee log_prepare_data_for_moses_lowercase

    ENTER_TAG = 'specialentertag' # 将数据中的\n换成ENTER_TAG，避免因为数据中的\n导致换行，从而导致生成的数据中code和comment不对齐
    load_keys = ['tok', 'comment']
    output_path = "/data/wanyao/work/baseline/corpus/training"

    # prefix = "issta20_c_aa0"
    # prefix = "fse20ruby100"
    # prefix = "fse20ruby100_v2"
    # prefix = "fse20ruby100_v3" # 100_small
    # prefix = "fse20ruby200"
    prefix = "fse20ruby100small_p0.01"

    if prefix == "fse20ruby100":
        path_root = "/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100/ruby/"
    elif prefix ==  "fse20ruby100_v2":
        path_root = "/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100/ruby/"
    elif prefix ==  "fse20ruby100_v3":
        path_root = "/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100_small/ruby/"
    elif prefix ==  "fse20ruby200":
        path_root = "/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/200/ruby/"
    elif prefix ==  "fse20ruby100small_p0.01" :
        path_root = "/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100_small/ruby_p0.01/"

    for mode in ['train','valid','test']:
        print("process ",mode)
        data = read_input(path_root, load_keys, mode)
        data = clean(data)
        dataset_type = prefix + '_' + mode
        output_code_name = "moses_data_lower." + dataset_type + ".code"
        output_com_name = "moses_data_lower." + dataset_type + ".com"
        output_code_path = os.path.join(output_path, output_code_name)
        output_com_path = os.path.join(output_path, output_com_name)
        print("output_code_path: ", output_code_path)
        print("output_com_path: ", output_com_path)
        save_data_step_one(data, output_code_path ,output_com_path)



    print("prefix: ", prefix)
    print("output_path: \n",output_path)


