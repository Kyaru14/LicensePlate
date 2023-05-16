# 介绍

- 电子科技大学--信息与通信工程学院课程--综合课程设计--车牌识别项目
- 本项目采用MTCNN+STN+LPRNet的方案
- MTCNN进行车牌检测及定位，STN进行仿射变换，LPRNet进行字符识别

# 使用

- 运行`main`对`data/images`和`data/ccpd_images`内的图片进行测试
- images是老师提供的2023测试集
- ccpd_images是ccpd数据集中的图片

# 训练

`train_Onet.py train_Pnet.py`和LPRNet文件夹内是训练代码

# development

- install pytorch at: https://pytorch.org/
- see requirements.txt for dependencies
- or run command: `pip install -r requirements.txt`