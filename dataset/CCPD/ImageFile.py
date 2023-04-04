import os.path
import re

class ImageFile:
    words_list = [
        "A", "B", "C", "D", "E",
        "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q",
        "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "0",
        "1", "2", "3", "4", "5",
        "6", "7", "8", "9"
    ]

    char_list = [
        "皖", "沪", "津", "渝", "冀",
        "晋", "蒙", "辽", "吉", "黑",
        "苏", "浙", "京", "闽", "赣",
        "鲁", "豫", "鄂", "湘", "粤",
        "桂", "琼", "川", "贵", "云",
        "西", "陕", "甘", "青", "宁",
        "新"
    ]
    def __init__(self, file_path):
        self.file_path = file_path
        self.basename = os.path.basename(file_path)
        pattern = re.compile('(.+)-(.+)-(.+)-(.+)-(.+)-(.+)-(.+).jpg')
        matcher = re.match(pattern, self.basename)
        pattern2 = re.compile('(\\d+)_(\\d+)')
        pattern3 = re.compile('(\\d+)&(\\d+)_(\\d+)&(\\d+)')
        matcher2 = re.match(pattern2, matcher.group(2))
        matcher3 = re.match(pattern3, matcher.group(3))
        self.horizon_angle = int(matcher2.group(1))
        self.vertical_angle = int(matcher2.group(2))
        self.points = [float(matcher3.group(1)), float(matcher3.group(2)), float(matcher3.group(3)), float(matcher3.group(4))]
        self.code = matcher.group(5).split('_')
        self.code = list(map(lambda x:int(x), self.code))
        self.label = ''
        for i, num in enumerate(self.code):
            if i == 0:
                self.label += ImageFile.char_list[num]
            else:
                self.label += ImageFile.words_list[num]

    def __repr__(self):
        return f'<class ImageFile> angle: {self.horizon_angle}, {self.vertical_angle} points: {self.points} label:{self.label}'

    @property
    def bbox(self):
        return self.points


if __name__ == '__main__':
    print(ImageFile(
        r'/License_Plate_Detection_Pytorch-master/ccpd_green/train/04-90_267-158&448_542&553-541&553_162&551_158&448_542&450-0_1_3_24_27_33_30_24-99-116.jpg'))
