import os

import json

import numpy as np

class YellowPlateDataset:
    def __init__(self, directory):
        paths = os.listdir(directory)
        json_paths = list(filter(lambda x: x.endswith('.json'), paths))
        self.files = []
        self.points = []
        for json_path in json_paths:
            with open(os.path.join(directory, json_path), mode='r', encoding='utf-8') as fp:
                param = json.load(fp)
            points = param['shapes'][0]['points']
            img_path = param['imagePath']
            points = [*points[0], *points[2]]
            self.files.append(os.path.join(directory, img_path))
            self.points.append(points)

    def __getitem__(self, item):
        return self.files[item], self.points[item]

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    YellowPlateDataset('../ccpd/yellow')
