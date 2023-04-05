import os

import json

import numpy as np

# target_dir = '../data/preprocessed/ccpd_train_split'

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
            points = np.asarray(points)
            x_mean = np.mean(points[:, 0])
            y_mean = np.mean(points[:, 1])
            points_filter = []
            for point in points:
                if point[0] < x_mean and point[1] < y_mean:
                    points_filter.append(point[0])
                    points_filter.append(point[1])
                if point[0] > x_mean and point[1] > y_mean:
                    points_filter.append(point[0])
                    points_filter.append(point[1])
            if len(points_filter) == 4:
                self.points.append(points_filter)
                self.files.append(os.path.join(directory, param['imagePath']))

    def __getitem__(self, item):
        return self.files[item], self.points[item]

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    YellowPlateDataset('../yellow')