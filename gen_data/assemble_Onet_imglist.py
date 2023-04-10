import os
import sys
sys.path.append(os.getcwd())
import assemble

mode = 'val'

if mode == 'train':
    onet_postive_file = 'data/train/pos_onet.txt'
    onet_part_file = 'data/train/part_onet.txt'
    onet_neg_file = 'data/train/neg_onet.txt'
    imglist_filename = 'data/train/imglist_onet.txt'
elif mode == 'val':
    onet_postive_file = 'data/val/pos_onet.txt'
    onet_part_file = 'data/val/part_onet.txt'
    onet_neg_file = 'data/val/neg_onet.txt'
    imglist_filename = 'data/val/imglist_onet.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)

    chose_count = assemble.assemble_data(imglist_filename, anno_list)
    print("ONet train annotation result file path:%s" % imglist_filename)