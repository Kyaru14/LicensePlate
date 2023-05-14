import json

from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        # for i in range(len(img_dir)):
        #     self.img_paths += [el for el in paths.list_images(img_dir[i])]
        for i in range(len(img_dir)):
            for path in paths.list_images(img_dir[i]):
                if (path[0] == '川'):
                    for tww in range(0, 20):
                        self.img_paths.append(path)
                self.img_paths.append(path)
        # random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
        with open('G:/plate/label.json', 'r') as fp:
            self.label_json = json.load(fp)
        self.len_json = len(os.listdir('G:/plate/clipped'))
        self.json_paths = os.listdir('G:/plate/clipped')

    def __len__(self):
        return len(self.img_paths) + 1 * self.len_json

    def __getitem__(self, index):
        if index < len(self.img_paths):
            filename = self.img_paths[index]
        else:
            filename = os.path.join('G:/plate/clipped', self.json_paths[(index - len(self.img_paths)) % self.len_json])
        # Image = cv2.imread(filename)
        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        if index < len(self.img_paths):
            basename = os.path.basename(filename)
            imgname, suffix = os.path.splitext(basename)
            imgname = imgname.split("-")[0].split("_")[0]
        else:
            imgname = self.label_json[self.json_paths[(index - len(self.img_paths)) % self.len_json]]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])

        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):

        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True