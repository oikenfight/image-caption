# -*- coding: utf-8 -*-

import argparse
import pickle as pickle
import json
import numpy as np
import time

word_counts = {}

word_ids = {
    '<S>': 0,
    '</S>': 1,
    '<UNK>': 2,
}
unknown = 2
min_word_count = 5

converted_annotations = {}
converted_image_ids = {}

output_dataset = {}


def prepare_converted_dataset(types: list, max_range=51) -> tuple:
    """

    :param list types: ['train', 'val', ...]
    :param int max_range: default 51
    :return:
        converted_annotations = {
            train: {[], [], ... , []},
            val: {[], [], ... , []}
        }
        converted_images = {...}
    ":rtype tuple:
    """
    converted_annotations = {
        k: {n: [] for n in range(1, max_range)} for k in types
    }
    converted_image_ids = {
        k: {n: [] for n in range(1, max_range)} for k in types
    }
    return converted_annotations, converted_image_ids


def count_words(input_dataset):
    """
    全アノテーションから key:出現単語, value:出現回数 の dict を作成する
    :param list input_dataset: [{'tokenized_caption': '私 は 犬 が 好き だ', 'id': int, 'image_id': int, 'caption': '私は犬が好きだ'}]
    :return: {'word1': int, 'word2': int, ... , 'wordN': int}
    :rtype: dict
    """
    global word_counts

    for data in input_dataset:
        for token in data['tokenized_caption'].split(' '):
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
    return word_counts


def create_word_ids_dict_with(word_counts: dict) -> dict:
    """
    文を word_id 集合に変換するために使用する dict ({key:単語, value: word_id, ...}) を作成する
    :param dict word_counts: {'word1': int, 'word2': int, ... , 'wordN': int}
    :return: {'word1': word_id, 'word2': word_id, ... , 'wordN': word_id}
    "rtype: dict
    """
    for word, count in word_counts.items():
        if count < min_word_count:
            continue
        word_ids[word] = len(word_ids)
    return word_ids


def words_to_ids(tokens: str) -> list:
    """
    形態素ごとにスペース区切りされた文字列を word_id 集合に変換する
    :param str tokens: '私 は 犬 が 好き だ'
    :return: [int, int, int, int, int]
    :rtype: list[int]
    """
    return [word_ids[token] if token in word_ids else unknown for token in tokens]


def create_converted_dataset(data_type: str, input_dataset: list) -> tuple:
    """
    文の長さ（単語数）ごとにデータをまとめる
    annotations と images のデータは配列順に対応する
    :param str data_type: 'train' or 'val' or ...
    :param list input_dataset: [{'tokenized_caption': '私 は 犬 が 好き だ', 'id': int, 'image_id': int, 'caption': '私は犬が好きだ'}, ...]
    :return:
        converted_sentences = {
            train: {
                1: [[], [], ... , []],        // sentence（word_id が集合してできた list）の集合。この場合 sentence（各list）の大きさは 1。
                2: [[], [], ... , []],        // sentence（word_id が集合してできた list）の集合。この場合 sentence（各list）の大きさは 2。
                .... ,
                10: [[], [], ... , []],       // sentence（word_id が集合してできた list）の集合。この場合 sentence（各list）の大きさは 10。
                11: [[], [], ... , []],       // sentence（word_id が集合してできた list）の集合。この場合 sentence（各list）の大きさは 11。
                ....
            },
            val: {...} ,
            ...
        }
        comverted_image_ids = {
            val: {
                1: [int, int, int],           // 単語数が1の画像の ID （int 型）の list
                2: [int, int, int],           // 単語数が2の画像の ID （int 型）の list
                .... ,
                10: [int, int, int],          // 単語数が10の画像の ID （int 型）の list
                11: [int, int, int],          // 単語数が11の画像の ID （int 型）の list
                ....
            },
            train: {...},
            ...
        }
    :rtype tuple(dict, dict):
    """
    global converted_annotations, converted_image_ids

    for annotation in input_dataset:
        tokens = annotation['tokenized_caption'].split(' ')
        if len(tokens) > 50:
            continue
        converted_annotations[data_type][len(tokens)].append(words_to_ids(tokens))
        converted_image_ids[data_type][len(tokens)].append(annotation['image_id'])
    return converted_annotations, converted_image_ids


def create_output_dataset() -> dict:
    """
    全てのデータセット内で使用される配列を numpy 型の list に変形し、まとめる
    :return: output dataset
    :rtype dict:
    """
    output_dataset['word_ids'] = word_ids
    output_dataset['annotations'] = {
        k: {n: np.array(sentences, dtype=np.int32) for n, sentences in v.items()} for k, v in converted_annotations.items()
    }
    output_dataset['images'] = {
        k: {n: np.array(image_ids, dtype=np.int32) for n, image_ids in v.items()} for k, v in converted_image_ids.items()
    }
    return output_dataset


def dump_to_pkl(output_pkl_file: str):
    """
    dump to pkl
    :param str output_pkl_file:
    :return:
    """
    with open(output_pkl_file, 'wb') as f:
        pickle.dump(output_dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON stair captions tokenized dataset to pkl')
    parser.add_argument('input_train', type=str, help='input train sentences JSON file path')
    parser.add_argument('input_val', type=str, help='input val sentences JSON file path')
    parser.add_argument('output', type=str, help='output dataset file path')
    args = parser.parse_args()

    print('~~~~~~~~~ step1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # データセット読み込み
    print('train file loading ...')
    with open(args.input_train) as f:
        input_train_dataset = json.load(f)
    print('val file loading ...')
    with open(args.input_val) as f:
        input_val_dataset = json.load(f)

    # データタイプごとの格納場所を生成
    print()
    print('~~~~~~~~~ step2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('prepare empty converted dataset.')
    converted_annotations, converted_image_ids = prepare_converted_dataset(['train', 'val'])

    # 全データから出現単語とその出現回数を集計
    print()
    print('~~~~~~~~~ step3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('count word with train dataset.')
    count_words(input_train_dataset['annotations'])
    print('count word with val dataset.')
    count_words(input_val_dataset['annotations'])

    # 出現回数が一定以下の単語を排除しつつ単語変換辞書の word_ids を作成
    print()
    print('~~~~~~~~~ step4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('create word_ids with word_counts.')
    word_ids = create_word_ids_dict_with(word_counts)

    # 文の長さごと（単語数）にデータをまとめる（同時に文をID集合に変換する）
    print()
    print('~~~~~~~~~ step5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('convert input_dataset to dataset arranged by number of words')
    create_converted_dataset('train', input_train_dataset['annotations'])
    create_converted_dataset('val', input_val_dataset['annotations'])

    # 全てのデータを numpy 型の list に変形する
    print()
    print('~~~~~~~~~ step6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('convert list to numpy list of each dataset, and create outpt dataset.')
    output_dataset = create_output_dataset()

    # print()
    # print('====== annotations ============================')
    # for data_type, dataset in output_dataset['annotations'].items():
    #     print('type: ' + str(data_type) + ', size:' + str(len(dataset)))
    #     for length_key, annotations in dataset.items():
    #         print('key(annotation length):' + str(length_key) + ', size:' + str(len(annotations)))
    #
    # print()
    # print('====== images ============================')
    # for data_type, dataset in output_dataset['images'].items():
    #     print('type: ' + str(data_type) + ', size:' + str(len(dataset)))
    #     for length_key, annotations in dataset.items():
    #         print('key(images length):' + str(length_key) + ', size:' + str(len(annotations)))

    print()
    print('~~~~~~~~~ step7 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('dumping now ...')
    dump_to_pkl(args.output)
