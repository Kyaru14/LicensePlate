import os
import random

import cv2
from tqdm import tqdm

from MTCNN_eval import *
from dataset.CCPD.CCPDDataset import CCPDDataset

train_dataset = CCPDDataset(directory='data/preprocessed/ccpd_train_split')
val_dataset = CCPDDataset(directory='data/preprocessed/ccpd_val_split')

count = 0
random.shuffle(train_dataset.files)
for image_file in tqdm(train_dataset.files, desc='generating lpr_train'):
    image = cv2.imread(image_file.file_path)
    x1, y1, x2, y2 = list(map(lambda x: int(x), image_file.points))
    image = image[y1:y2, x1:x2]
    cv2.imencode('.jpg', image)[1].tofile(f'data/preprocessed/lpr_train/{image_file.label}.jpg')

for image_file in tqdm(val_dataset.files, desc='generating lpr_val'):
    image = cv2.imread(image_file.file_path)
    x1, y1, x2, y2 = list(map(lambda x: int(x), image_file.points))
    image = image[y1:y2, x1:x2]
    cv2.imencode('.jpg', image)[1].tofile(f'data/preprocessed/lpr_train/{image_file.label}.jpg')
