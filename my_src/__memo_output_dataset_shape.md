## pkl に dump するために必要な format

```angular2html

output_word_ids_dataset = {
    'word1': key,
    'word2': key,
    'word3': key,
    ....
}

// output_sentences_dataset と output_sentences_dataset は完全に対応している

output_sentences_dataset = {
    val: {
        1: [[], [], ... , []],            // sentence（word_id が集合してできた Array）の集合。この場合 sentence の大きさは 1。
        2: [[], [], ... , []],            // sentence（word_id が集合してできた Array）の集合。この場合 sentence の大きさは 2。
        .... ,
        10: [[], [], ... , []],           // sentence（word_id が集合してできた Array）の集合。この場合 sentence の大きさは 10。
        11: [[], [], ... , []],           // sentence（word_id が集合してできた Array）の集合。この場合 sentence の大きさは 11。
        ....
    },
    train: {},                            // 同様
    restval: {},                          // 同様
    test: {},                             // 同様
}

output_images_dataset = {
    val: {
        1: [int, int, int],               // 単語数が1の画像の ID （int 型）の Array
        2: [int, int, int],               // 単語数が2の画像の ID （int 型）の Array
        .... ,
        10: [int, int, int],              // 単語数が10の画像の ID （int 型）の Array
        11: [int, int, int],              // 単語数が11の画像の ID （int 型）の Array
        ....
    },
    train: {},                            // 同様
    restval: {},                          // 同様
    test: {},                             // 同様
}

output_dataset = {
    output_word_ids_dataset,
    output_sentences_dataset,
    output_images_dataset,
}

```