import os
import sys
sys.path.append(os.getcwd())
import assemble

mode = 'val'

if mode == 'train':
    pnet_postive_file = 'data/train/pos_pnet.txt'
    pnet_part_file = 'data/train/part_pnet.txt'
    pnet_neg_file = 'data/train/neg_pnet.txt'
    imglist_filename = 'data/train/imglist_pnet.txt'
elif mode == 'val':
    pnet_postive_file = 'data/val/pos_pnet.txt'
    pnet_part_file = 'data/val/part_pnet.txt'
    pnet_neg_file = 'data/val/neg_pnet.txt'
    imglist_filename = 'data/val/imglist_pnet.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename, anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
