import argparse
import json

parser = argparse.ArgumentParser(description='Checking Json file')

parser.add_argument('--json_file_path', '-p', required=True, type=str,
                    help='input json file path you want to read')

parser.add_argument('--read_num', '-n', required=True, type=int,
                    help='How far to read')

args = parser.parse_args()

print(args.json_file_path)
print(args.read_num)

with open(args.json_file_path, 'r') as json_file:
    dataset = json.load(json_file)

    print(len(dataset))
    for data in dataset:
        print(data)

    print('image len and type')
    print(len(dataset['images']))
    print(type(dataset['images']))
    # print(dataset['images'])
    # print(dataset['dataset'])

    cnt = 0
    for data in dataset['images']:
        print(data)
        print()
        if cnt == args.read_num:
            break
        cnt += 1

    # cnt = 0
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for data in dataset['images']:
        if data['imgid'] == 0:
            print(data)

