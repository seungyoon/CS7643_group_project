import generators
import sys

def data_to_csv(filename, example):
    input, target, taskid = example
    input = input.astype(int)
    with open(filename, 'w+') as f:
        sys.stdout = f
        print("input,target")
        length = len(input)

        for i in range(length):
            for num in input[i][0]:
                print(num, end=' ')
    
            print(',', end=' ')

            for num in target[i]:
                print(num, end=' ')

            print("")

for data_size in ['small', 'middle', 'large']:
    if data_size == 'small':
        length, test_length = 8, 8
        train_data_size, val_test_data_size = 20480, 2560
    elif data_size == 'middle':
        length, test_length = 40, 40
        train_data_size, val_test_data_size = 20480, 2560
    elif data_size == 'large':
        length, test_length = 40, 400
        train_data_size, val_test_data_size = 204800, 25600
    for task in ['badd', 'add', 'rev', 'scopy']:  # binary add, add, reverse, copy
        # train
        example = generators.generators[task].get_batch(length, train_data_size)
        data_to_csv("data/" + data_size + '/' + task + '-train.csv', example)
        # validataion
        example = generators.generators[task].get_batch(length, val_test_data_size)
        data_to_csv("data/" + data_size + '/' + task + '-validation.csv', example)
        # test
        example = generators.generators[task].get_batch(test_length, val_test_data_size)
        data_to_csv("data/" + data_size + '/' + task + '-test.csv', example)
