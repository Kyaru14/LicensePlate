import sys
sys.path.append('../..')
import cv2
import os
from util import *
from dataset.CCPD import CCPDDataset
from dataset.YellowPlateDataset import YellowPlateDataset

mode = 'val'

if mode == 'train':
    img_dir = "../data/preprocessed/ccpd_train_split"
    yellow_dir = "../data/preprocessed/ccpd_train_split/yellow"
    pos_save_dir = "data/train/pnet/positive"
    part_save_dir = "data/train/pnet/part"
    neg_save_dir = "data/train/pnet/negative"

    # store labels of positive, negative, part images
    f1 = open(os.path.join('../data/train', 'pos_pnet.txt'), 'w')
    f2 = open(os.path.join('../data/train', 'neg_pnet.txt'), 'w')
    f3 = open(os.path.join('../data/train', 'part_pnet.txt'), 'w')
elif mode == 'val':
    img_dir = "../data/preprocessed/ccpd_val_split"
    yellow_dir = "../data/preprocessed/ccpd_val_split/yellow"
    pos_save_dir = "data/val/pnet/positive"
    part_save_dir = "data/val/pnet/part"
    neg_save_dir = "data/val/pnet/negative"

    # store labels of positive, negative, part images
    f1 = open(os.path.join('data/val', 'pos_pnet.txt'), 'w')
    f2 = open(os.path.join('data/val', 'neg_pnet.txt'), 'w')
    f3 = open(os.path.join('data/val', 'part_pnet.txt'), 'w')

if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

dataset = CCPDDataset.CCPDDataset(directory=img_dir)
yellow_dataset = YellowPlateDataset(directory=yellow_dir)

p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # dont care
idx = 0
image_list = []
for file in dataset.files:
    image_list.append((file.file_path, file.bbox))
for path, points in yellow_dataset:
    image_list.append((path, points))
print(f'{len(image_list)} in total')
for path, points in image_list:
    im_path = path

    boxes = np.zeros((1, 4), dtype=np.int32)
    boxes[0, :] = points

    img = cv2.imread(im_path)
    idx += 1

    height, width, channel = img.shape
    
    neg_num = 0
    while neg_num < 35:
        size_x = np.random.randint(47, min(width, height) / 2)
        size_y = np.random.randint(12, min(width, height) / 2)
        nx = np.random.randint(0, width - size_x)
        ny = np.random.randint(0, height - size_y)
        crop_box = np.array([nx, ny, nx + size_x, ny + size_y])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny: ny + size_y, nx: nx + size_x, :]
        resized_im = cv2.resize(cropped_im, (47, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
            
    for box in boxes:
        # box (x_left, y_top, w, h)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if int(w * 0.2) <= 0 or int(h * 0.2) <= 0:
            continue

        # generate negative examples that have overlap with gt
        for i in range(5):
            size_x = np.random.randint(47, min(width, height) / 2)
            size_y = np.random.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = np.random.randint(max(-size_x, -x1), w)
            delta_y = np.random.randint(max(-size_y, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)

            if nx1 + size_x > width or ny1 + size_y > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size_x, ny1 + size_y])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size_y, nx1: nx1 + size_x, :]
            resized_im = cv2.resize(cropped_im, (47, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
        # generate positive examples and part faces
        for i in range(20):
            size_x = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            size_y = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size_x / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size_y / 2, 0)
            nx2 = nx1 + size_x
            ny2 = ny1 + size_y

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size_x)
            offset_y1 = (y1 - ny1) / float(size_y)
            offset_x2 = (x2 - nx2) / float(size_x)
            offset_y2 = (y2 - ny2) / float(size_y)

            cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
            resized_im = cv2.resize(cropped_im, (47, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4 and d_idx < 1.2*p_idx + 1:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1



    print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
        