import os.path

import torch
import torch.utils.data as data
import cv2
import numpy as np

from dataset.CCPD.ImageFile import ImageFile


class TrainFolder(data.Dataset):
    def __init__(self, directory: str):
        self.directory = os.path.join(directory)
        self.files = []
        for filename in os.listdir(self.directory):
            self.files.append(ImageFile(os.path.join(self.directory, filename)))

    def __getitem__(self, item):
        image_file = self.files[item]
        img = cv2.imread(image_file.file_path)
        img = img[:, :, ::-1]
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)

        label = 1
        bbox_target = np.array(image_file.points)

        image = {'input_img': input_img, 'label': label, 'bbox_target': bbox_target}

        return image
    def __len__(self):
        return len(self.files)


class TestFolder(TrainFolder):
    def __init__(self, *, directory: str, mode: str = 'test'):
        super().__init__(directory=directory)


class ValFolder(TrainFolder):
    def __init__(self, *, directory: str, mode: str = 'val'):
        super().__init__(directory=directory)

if __name__ == '__main__':
    print(ValFolder(directory='../License_Plate_Detection_Pytorch-master/ccpd_green').files)
