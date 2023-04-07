import os

import cv2
from tqdm import tqdm

from MTCNN import *
from dataset.CCPD.CCPDDataset import CCPDDataset

train_dataset = CCPDDataset(directory='data/preprocessed/ccpd_train_split')
val_dataset = CCPDDataset(directory='data/preprocessed/ccpd_val_split')

# print(os.listdir('data/preprocessed/lpr_train')[0].encode('gbk').decode('utf-8'))

for image_file in tqdm(train_dataset.files, desc='generating lpr_train'):
    image = cv2.imread(image_file.file_path)
    x1, y1, x2, y2 = list(map(lambda x: int(x), image_file.points))
    image = image[y1:y2, x1:x2]
    cv2.imwrite(f'data/preprocessed/lpr_train/{image_file.label}.jpg', image)

for image_file in tqdm(val_dataset.files, desc='generating lpr_val'):
    image = cv2.imread(image_file.file_path)
    x1, y1, x2, y2 = list(map(lambda x: int(x), image_file.points))
    image = image[y1:y2, x1:x2]
    cv2.imwrite(f'data/preprocessed/lpr_test/{image_file.label}.jpg', image)
