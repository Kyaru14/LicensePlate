import os.path

import torch
import torch.utils.data as data
import cv2
import numpy as np

from dataset.CCPD.ImageFile import ImageFile


class CCPDDataset(data.Dataset):
    def __init__(self, directory: str):
        self.directory = os.path.join(directory)
        self.files = []
        for filename in os.listdir(self.directory):
            path = os.path.join(self.directory, filename)
            if os.path.isfile(path):
                self.files.append(ImageFile(path))

    def __getitem__(self, item):
        image_file = self.files[item]
        img = cv2.imread(image_file.file_path)
        input_img = img[:, :, ::-1]
        input_img = np.asarray(input_img, 'float32')
        input_img = input_img.transpose((2, 0, 1))
        input_img = (input_img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(input_img)

        label = 1
        bbox_target = np.array(image_file.points)

        return input_img, label, bbox_target
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    print(CCPDDataset(directory='../License_Plate_Detection_Pytorch-master/ccpd_green').files)
