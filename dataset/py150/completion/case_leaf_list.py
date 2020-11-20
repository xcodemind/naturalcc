import os
import ujson


def get_code_tokens(line):
    """get DFS leaf node of ast"""
    ast = ujson.loads(line)
    code_tokens = [node["value"] for node in ast if "value" in node]
    return code_tokens


train_file = "~/.ncc/py150/raw/python100k_train.json"
test_file = "~/.ncc/py150/raw/python50k_eval.json"

for file in [train_file, test_file]:
    file = os.path.expanduser(file)
    with open(file, 'r') as reader, open(file + '.leaf', 'w') as writer:
        for line in reader:
            leaf_tokens = get_code_tokens(line)
            print(ujson.dumps(leaf_tokens), file=writer)
