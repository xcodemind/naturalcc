
import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        aa = json_file.readlines()[0]
        print("len(aa): ",len(aa))
        output = json.loads(aa)
    return output

import ujson

# path = """/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result/summarization/mm2seq/maml_java-javascript-php-python_ruby_Adam\(0.0004\)_SGD\(0.001\)-10-1.new/train_maml/tok8path-bs128-Adam\(0.0004\)-mSGD\(0.001\)-EPOCH10-1-3000.pred"""
path = 'maml.json'

# b=load_json(path)
def load_json_by_line(path ):
    data=[]
    with open(path , "r", encoding="utf-8") as reader:
        # data = ujson.loads(reader.read())
        while True :
            line =     reader.readline().strip()
            if  line:
                dat  = ujson.loads(line )
                data.append(dat)
            else:
                break
    return data

print("len(data): ",len(data  ))
# print("b.keys(): ",b.keys()   )
# print("len(b['tok_len']): ",len(b['tok_len']))
# print("len(b['comment_len']): ",len(b['comment_len']))
# print("len(b['ast_len']): ",len(b['ast_len']))
# print("len(b['rougel']): ",len(b['rougel']))

#dict_keys(['src_comment', 'trg_comment', 'pred_comment', 'src_code', 'tok_len', 'comment_len', 'ast_len',
# 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougel', 'cider'])

# 'tok_len'
# 'comment_len'
# 'ast_len'
#
# 'bleu1'
# 'meteor'
# 'rougel'
# 'cider'