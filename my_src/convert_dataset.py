# -*- coding: utf-8 -*-

import argparse
import pickle as pickle
import json
import numpy as np

parser = argparse.ArgumentParser(description='Convert JSON dataset to pkl')
parser.add_argument('input', type=str,
                    help='input sentences JSON file path')
parser.add_argument('output', type=str,
                    help='output dataset file path')
args = parser.parse_args()

print('file loading ...')
with open(args.input) as f:
    input_dataset = json.load(f)

word_ids = {
    '<S>': 0,
    '</S>': 1,
    '<UNK>': 2,
}
unknown = 2
min_word_count = 5

converted_sentences = {
    k: {n: [] for n in range(1, 51)} for k in ['train', 'val', 'test', 'restval']
}
converted_image_ids = {
    k: {n: [] for n in range(1, 51)} for k in ['train', 'val', 'test', 'restval']
}


def words_to_ids(tokens):
    return [ word_ids[token] if token in word_ids else unknown for token in tokens]


# treat words as <UNK> that appear few times
word_counts = {}
for image in input_dataset['images']:
    data_type = image['split']
    if data_type != 'train':
        continue
    for sentence in image['sentences']:
        for token in sentence['tokens']:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
for word, count in word_counts.items():
    if count < min_word_count:
        continue
    word_ids[word] = len(word_ids)

for image in input_dataset['images']:
    image_id = image['imgid']
    data_type = image['split']
    for sentence in image['sentences']:
        tokens = sentence['tokens']
        converted_sentences[data_type][len(tokens)].append(words_to_ids(sentence['tokens']))
        converted_image_ids[data_type][len(tokens)].append(image_id)

for k, v in converted_sentences.items():
    print(str(k) + ': ' + str(len(v)))
    for n, sentences in v.items():
        print(str(n) + ': ' + str(len(sentences)))
        if sentences:
            print('sentence length:' + str(len(sentences[0])))

            print('~~~~~ sample sentence ~~~~~~~~~')
            sentence_count = 0
            for sentence in sentences:
                print(sentence)
                if sentence_count > 5:
                    break
                sentence_count += 1

output_dataset = {}
output_dataset['sentences'] = {
    k: {n: np.array(sentences, dtype=np.int32) for n, sentences in v.items()} for k, v in converted_sentences.items()
}
output_dataset['word_ids'] = word_ids
output_dataset['images'] = {
    k: {n: np.array(image_ids, dtype=np.int32) for n, image_ids in v.items()} for k, v in converted_image_ids.items()
}


word_ids_data = output_dataset['word_ids']
sentences_data = output_dataset['sentences']
images_data = output_dataset['images']

print('word_ids shape')
print(word_ids)

print(len(sentences_data))
for data in sentences_data:
    print(data)

print('sentence_data shape')
for key, data in sentences_data['test'].items():
    print('key: ' + str(key) + ', sentence_num: ' + str(len(data)))

print()
print('image_data shape')
for key, data in images_data['test'].items():
    print('key: ' + str(key) + ', image_id: ' + str(len(data)))



# print(sentences_data['val'][7])

# for i in range(10):
#     print('========' + str(i) + '======================')
#     print('sentences:')
#     print(sentences_data['val'])
#     print('images:')
#     print(images_data['val'])
#     print()

# print('dumping now ...')
# with open(args.output, 'wb') as f:
#     pickle.dump(output_dataset, f, pickle.HIGHEST_PROTOCOL)
