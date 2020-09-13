import os

data_path = os.path.expanduser('~/.ncc/augmented_javascript/type_prediction/raw')
output_path = os.path.expanduser('~/.ncc/augmented_javascript/type_prediction/data-raw')

if not os.path.exists(output_path):
    os.makedirs(output_path)

def cast_file(file_name):
    with open(file_name, 'r') as input_file, open(os.path.join(output_path, 'target.dict.txt'), 'w') as output_file:
        for line in input_file.readlines():
            output_file.write(line.strip('\n') + ' ' + '1' + '\n')


if __name__ == '__main__':
    dict_file = os.path.join(data_path, 'target_wl')

    cast_file(dict_file)
