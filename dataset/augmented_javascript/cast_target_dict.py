import os

data_path = '/home/wanyao/.ncc/augmented_javascript/contracode/types'

def cast_file(file_name):
    with open(file_name, 'r') as input_file, open(os.path.join(data_path, 'target.dict.txt'), 'w') as output_file:
        for line in input_file.readlines():
            output_file.write(line.strip('\n') + ' ' + '1' + '\n')


if __name__ == '__main__':
    dict_file = os.path.join(data_path, 'target_wl')

    cast_file(dict_file)
