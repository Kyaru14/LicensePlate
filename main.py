import json
import os

import torch
import cv2
import numpy as np

import MTCNN_eval
import model.STN as STN
import model.LPRNet as LPRNet
import dataset.LPRDataset as LPRDataLoader
import LPRNet_eval

model_paths = {
    'p_net_path': 'data/net/pnet.weights',
    'o_net_path': 'data/net/onet.weights',
    'stn_net_path': 'data/net/Final_STN_model(1).pth',
    'lpr_net_path': 'data/net/Final_LPRNet_model(1).pth'
}
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

STNet = STN.STNet()
STNet.load_state_dict(torch.load(model_paths['stn_net_path']))
STNet = STNet.to(device)

lprnet = LPRNet.build_lprnet(lpr_max_len=8, phase=False, class_num=len(LPRDataLoader.CHARS), dropout_rate=0)
lprnet = lprnet.to(device)


def detect_plate(image_path, *, scale, minimum_lp, model_paths, device):
    image = cv2.imread(image_path)
    if scale == 'auto':
        w, h = len(image[0]), len(image)
        E = 1600
        scale = (E / h + E / w) / 2
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    bboxes = MTCNN_eval.create_mtcnn_net(image, minimum_lp, device, p_model_path=model_paths['p_net_path'],
                                         o_model_path=model_paths['o_net_path'])

    result = []
    area = 0
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :4]
        x1, y1, x2, y2 = list(map(lambda x: int(x), bbox))
        plate = cv2.resize(image[y1:y2, x1:x2], (94, 24), interpolation=cv2.INTER_CUBIC)

        plate = plate.astype('float32')
        plate -= 127.5
        plate *= 0.0078125
        plate = np.transpose(plate, (2, 0, 1))
        plate = torch.tensor(plate).to(device)
        plate = plate.unsqueeze(0)

        plate = STNet(plate)

        lprnet.load_state_dict(torch.load(model_paths['lpr_net_path']))

        label = LPRNet_eval.Greedy_Decode_Eval_1image(lprnet, plate)

        if abs(x1 - x2) * abs(y1 - y2) > area:
            area = abs(x1 - x2) * abs(y1 - y2)
            cv2.imencode('.jpg', image[y1: y2, x1: x2])[1].tofile(f'data/clipped/{os.path.basename(image_path)}')
            cv2.imencode('.jpg',
                         (plate.squeeze().detach().cpu().numpy() * 128 + 127.5).transpose(1, 2, 0).astype('int'))[
                1].tofile(f'data/transformed/{os.path.basename(image_path)}')
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        image = LPRNet_eval.cv2ImgAddText(image, label, (x1, y1 - 30), textColor=(100, 255, 100), textSize=24)
        result.append(((x1, y1, x2, y2), label))

    return result, image


if __name__ == '__main__':
    with open('data/label.json', 'r') as fp:
        label = json.load(fp)
    dir = 'data/images'
    metric = [0., len(os.listdir(dir))]
    for i, path in enumerate(os.listdir(dir)):
        print(f'{i}/{metric[1]}', path)
        result, image = detect_plate(os.path.join(dir, path), scale='auto', minimum_lp=(50, 15),
                                     model_paths=model_paths,
                                     device=device)
        cv2.imencode('.jpg', image)[1].tofile(f'data/result/{path}')
        print('expected:', label[path])
        for r in result:
            print('predict:', r[1])
            if r[1] == label[path]:
                metric[0] += 1
                break
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print(f'Accuracy: {metric[0] / metric[1] * 100:.6f}%')
