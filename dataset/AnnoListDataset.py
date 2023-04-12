import os.path

import torch
import torch.utils.data as data
import cv2
import numpy as np

from tqdm import tqdm

class AnnoListDataset(data.Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.preloaded_data = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.preloaded_data is None:
            return self.gen_sample(self.img_files[index % len(self.img_files)].strip().split(' '))
        else:
            return self.preloaded_data[index]

    def gen_sample(self, img_file_annotation):
        annotation = img_file_annotation

        img = cv2.imread(annotation[0])
        img = img[:, :, ::-1]
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)

        label = int(annotation[1])
        bbox_target = np.zeros((4,))

        if len(annotation[2:]) == 4:
            bbox_target = np.array(annotation[2:6]).astype(float)

        # sample = {'input_img': input_img, 'label': label, 'bbox_target': bbox_target}

        return input_img, label, bbox_target

    def __enter__(self):
        self.preloaded_data = []
        for annotation in tqdm(self.img_files, desc='Preloading images'):
            self.preloaded_data.append(self.gen_sample(annotation.strip().split(' ')))

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.preloaded_data
        self.preloaded_data = None
