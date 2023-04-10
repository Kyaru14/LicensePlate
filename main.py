import os

import torch
import cv2
import numpy as np
from tqdm import tqdm

import MTCNN_eval
import model.STN as STN
import model.LPRNet as LPRNet
import dataset.LPRDataset as LPRDataLoader
import LPRNet_eval


def detect_plate(image_path, *, scale, minimum_lp, model_paths, device):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    bboxes = MTCNN_eval.create_mtcnn_net(image, minimum_lp, device, p_model_path=model_paths['p_net_path'],
                                         o_model_path=model_paths['o_net_path'])

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :4]
        x1, y1, x2, y2 = list(map(lambda x: int(x), bbox))
        plate = cv2.resize(image[y1:y2, x1:x2], (94, 24), interpolation=cv2.INTER_CUBIC)

        # cv2.imshow('image', plate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        plate = plate.astype('float32')
        plate -= 127.5
        plate *= 0.0078125
        plate = np.transpose(plate, (2, 0, 1))
        plate = torch.tensor(plate).to(device)
        plate = plate.unsqueeze(0)

        STNet = STN.STNet()
        STNet.load_state_dict(torch.load(model_paths['stn_net_path']))
        STNet = STNet.to(device)
        plate = STNet(plate)

        # plate_n = plate.squeeze().detach().cpu().numpy()
        # cv2.imshow('image', ((plate_n.transpose(1, 2, 0)) * 128))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        lprnet = LPRNet.build_lprnet(lpr_max_len=8, phase=False, class_num=len(LPRDataLoader.CHARS), dropout_rate=0)
        lprnet = lprnet.to(device)
        lprnet.load_state_dict(torch.load(model_paths['lpr_net_path']))

        label = LPRNet_eval.Greedy_Decode_Eval_1image(lprnet, plate)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        image = LPRNet_eval.cv2ImgAddText(image, label, (x1, y1 - 24), textColor=(100, 255, 100), textSize=24)

    image = cv2.resize(image, (0, 0), fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
    return image


if __name__ == '__main__':
    model_paths = {
        'p_net_path': 'data/net/pnet.weights',
        'o_net_path': 'data/net/onet.weights',
        'stn_net_path': 'data/net/stn.weights',
        'lpr_net_path': 'data/net/lprnet.weights'
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for path in tqdm(os.listdir('data/images')):
        image = detect_plate(os.path.join('data/images', path), scale=0.6, minimum_lp=(50, 15), model_paths=model_paths,
                             device=device)
        cv2.imencode('.jpg', image)[1].tofile(f'data/result/{path}')
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
