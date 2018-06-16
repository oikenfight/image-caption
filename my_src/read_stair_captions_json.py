import argparse
import json
import time

parser = argparse.ArgumentParser(description='Checking stair captions json file')

parser.add_argument('--json_file_path', '-p', required=True, type=str,
                    help='input json file path you want to read')

parser.add_argument('--read_num', '-n', required=True, type=int,
                    help='How far to read')

args = parser.parse_args()

print("file_path: " + args.json_file_path)
print("read_num: " + str(args.read_num))

with open(args.json_file_path, 'r') as json_file:
    read_data = json.load(json_file)

    print('images len: ' + str(len(read_data['images'])))
    print('annotations len: ' + str(len(read_data['annotations'])))

    for data in read_data:
        print(data)

    print('~~~~~~~~~ images ~~~~~~~~~~~~~~~~~~~~~~~~~~')
    images = read_data['images']
    for i in range(args.read_num):
        print('--------' + str(i) + ' ----------------------')
        print(images[i])
        print()

    print('~~~~~~~~~ annotations ~~~~~~~~~~~~~~~~~~~~~~~~~~')
    annotations = read_data['annotations']
    for i in range(args.read_num):
        print('--------' + str(i) + ' ----------------------')
        print(annotations[i])
        print()

    for annotation in annotations:
        if annotation['image_id'] == 179765:
            print(annotation)

    print('~~~~~~~~~ match images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for image in images:
        if image['id'] == 203312:
            print(image)
