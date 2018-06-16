## coco_dataset_format

```
dataset = {
    images: [
        {
            cocoid: int,
            finename: 'COCO_val2014_000000391895.jpg',
            split: 'test',
            filepath: 'val2014',
            sentids: [int, int, int, int, int],
            sentences: [
                {
                    tokens: ["単語", "単語", ..., "単語", "単語"],
                    sentid: int,
                    imgid: int,
                    raw: '元の文',
                },
                {},
                .... ,
                {},
            ],
        },
        {},
        ...
        {},
        {},
    ]
    dataset: 'coco'
}
```