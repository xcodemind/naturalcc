import os
import ujson

code_files = os.path.expanduser('~/.ncc/python_wan/flatten/test.code')

with open(code_files, 'r') as reader:
    codes = [ujson.loads(line) for line in reader]

predict_file = os.path.join(os.path.dirname(__file__), './predict.json')
case_file = os.path.join(os.path.dirname(__file__), './case.txt')
with open(predict_file, 'r') as reader, open(case_file, 'w') as writer:
    for line in reader:
        line = ujson.loads(line)
        if line['bleu'] == 1.:
            print(ujson.dumps([codes[line['id']], line['references']]), file=writer)
