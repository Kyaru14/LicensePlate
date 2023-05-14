import os
import random

import cv2
from tqdm import tqdm

from MTCNN_eval import *
from dataset.YellowPlateDataset import YellowPlateDataset

train_dataset = YellowPlateDataset(directory='data/preprocessed/ccpd_train_split')
val_dataset = YellowPlateDataset(directory='data/preprocessed/ccpd_val_split')

for image_file in tqdm(train_dataset.files, desc='generating lpr_train_yellow'):
    image = cv2.imread(image_file.file_path)
    x1, y1, x2, y2 = list(map(lambda x: int(x), image_file.points))
    image = image[y1:y2, x1:x2]
    cv2.imencode('.jpg', image)[1].tofile(f'data/preprocessed/lpr_train/{os.path.basename(image_file.file_path)}')
    cv2.imencode('.jpg', image)[1].tofile(f'data/preprocessed/lpr_train/{os.path.basename(image_file.file_path)}')

for image_file in tqdm(val_dataset.files, desc='generating lpr_val_yellow'):
    image = cv2.imread(image_file.file_path)
    x1, y1, x2, y2 = list(map(lambda x: int(x), image_file.points))
    image = image[y1:y2, x1:x2]
    cv2.imencode('.jpg', image)[1].tofile(f'data/preprocessed/lpr_test/{os.path.basename(image_file.file_path)}')
