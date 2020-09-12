import os

data_path = os.path.expanduser('~/.ncc/augmented_javascript/type_prediction/raw')
output_path = os.path.expanduser('~/.ncc/augmented_javascript/type_prediction/data-raw')

def cast_file(file_name, mode, src, tgt):
    with open(file_name, 'r') as input_file, open(os.path.join(output_path, '{}.{}'.format(mode, src)), 'w') as code_file, \
            open(os.path.join(output_path, '{}.{}'.format(mode, tgt)), 'w') as type_file:
        for line in input_file.readlines():
            code_file.write(line.split('\t')[0] + '\n')
            type_file.write(line.split('\t')[1])


if __name__ == '__main__':
    train_file = os.path.join(data_path, 'train_nounk.txt')
    valid_file = os.path.join(data_path, 'valid_nounk.txt')
    test_file = os.path.join(data_path, 'test_nounk.txt')
    test_file_filtered = os.path.join(data_path, 'test_projects_gold_filtered.json')

    cast_file(train_file, 'train', 'code', 'type')
    cast_file(valid_file, 'valid', 'code', 'type')
    cast_file(test_file, 'test', 'code', 'type')
    cast_file(test_file_filtered, 'test', 'code_filtered', 'type_filtered')
