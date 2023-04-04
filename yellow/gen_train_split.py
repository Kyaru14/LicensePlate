import os

import json

target_dir = '../data/preprocessed/ccpd_train_split'
files = os.listdir('.')
json_paths = list(filter(lambda x: x.endswith('.json'), files))

for json_path in json_paths:
    with open(json_path, mode='r', encoding='utf-8') as fp:
        param = json.load(fp)
    image_path = param['imagePath']
    print(param)
    points = [*param['shapes'][0]['points'][0], *param['shapes'][0]['points'][2]]
    print(points)
    exit()
