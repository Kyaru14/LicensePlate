import sys
sys.path.append('../../..')
import cv2
import os
from util import *
import torch
from MTCNN_eval import create_mtcnn_net
from dataset.CCPD import CCPDDataset
from dataset.YellowPlateDataset import YellowPlateDataset

mode = 'val'

if mode == 'train':
    img_dir = "../data/preprocessed/ccpd_train_split"
    yellow_dir = "../data/preprocessed/ccpd_train_split/yellow"
    pos_save_dir = "data/train/onet/positive"
    part_save_dir = "data/train/onet/part"
    neg_save_dir = "data/train/onet/negative"

    # store labels of positive, negative, part images
    f1 = open(os.path.join('../data/train', 'pos_onet.txt'), 'w')
    f2 = open(os.path.join('../data/train', 'neg_onet.txt'), 'w')
    f3 = open(os.path.join('../data/train', 'part_onet.txt'), 'w')
elif mode == 'val':
    img_dir = "../data/preprocessed/ccpd_val_split"
    yellow_dir = "../data/preprocessed/ccpd_val_split/yellow"
    pos_save_dir = "data/val/onet/positive"
    part_save_dir = "data/val/onet/part"
    neg_save_dir = "data/val/onet/negative"

    # store labels of positive, negative, part images
    f1 = open(os.path.join('data/val', 'pos_onet.txt'), 'w')
    f2 = open(os.path.join('data/val', 'neg_onet.txt'), 'w')
    f3 = open(os.path.join('data/val', 'part_onet.txt'), 'w')

if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_size = (94, 24)

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

    image = cv2.imread(im_path)

    bboxes = create_mtcnn_net(image, (50, 15), device, p_model_path='../data/images/pnet.weights', o_model_path=None)
    dets = np.round(bboxes[:, 0:4])

    if dets.shape[0] == 0:
        continue

    img = cv2.imread(im_path)
    idx += 1

    height, width, channel = img.shape

    for box in dets:
        x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
        width = x_right - x_left + 1
        height = y_bottom - y_top + 1

        # ignore box that is too small or beyond image border
        if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
            continue

        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, boxes)
        cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
        resized_im = cv2.resize(cropped_im, image_size, interpolation=cv2.INTER_LINEAR)

        # save negative images and write label
        if np.max(Iou) < 0.3 and n_idx < 3.2*p_idx+1:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
        else:
            # find gt_box with the highest iou
            idx_Iou = np.argmax(Iou)
            assigned_gt = boxes[idx_Iou]
            x1, y1, x2, y2 = assigned_gt

            # compute bbox reg label
            offset_x1 = (x1 - x_left) / float(width)
            offset_y1 = (y1 - y_top) / float(height)
            offset_x2 = (x2 - x_right) / float(width)
            offset_y2 = (y2 - y_bottom) / float(height)

            # save positive and part-face images and write labels
            if np.max(Iou) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

            elif np.max(Iou) >= 0.4 and d_idx < 1.2*p_idx + 1:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

    print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()







