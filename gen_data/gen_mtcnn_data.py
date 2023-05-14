import os
import random

from shutil import copyfile

from tqdm import tqdm

ccpd2019_dir = '../ccpd/CCPD2019'

dirs = ['ccpd_base', 'ccpd_rotate', 'ccpd_challenge']
train_num_files = [10000, 5000, 5000]
val_num_files = [2000, 1000, 1000]

for i, directory in enumerate(dirs):
    directory = os.path.join(ccpd2019_dir, directory)
    image_paths = os.listdir(directory)
    random.shuffle(image_paths)
    count = 0
    for _ in tqdm(range(train_num_files[i]), desc=directory):
        copyfile(src=os.path.join(directory, image_paths[count]),
                 dst=os.path.join("../data/preprocessed/ccpd_train_split", image_paths[count]))
        count += 1
    for _ in tqdm(range(val_num_files[i]), desc=directory):
        copyfile(src=os.path.join(directory, image_paths[count]),
                 dst=os.path.join("../data/preprocessed/ccpd_val_split", image_paths[count]))
        count += 1
