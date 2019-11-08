import os
import cv2
import numpy as np
import time
from collections import Counter


def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i+batch_size, data_num), ...]


def load_image(pics_path, image_size, name_is_folder=1):
    print('pic reading %s' % pics_path)
    if os.path.isdir(pics_path):
        paths = list(os.listdir(pics_path))
        if '.DS_Store' in paths:  # 去掉不为文件夹格式的mac os系统文件
            paths.remove('.DS_Store')
    else:
        paths = [pics_path]
    images = []
    fns = []
    for peo in paths:
        peo_i = 0
        peo_pics = list(os.listdir(pics_path + '/' + peo + '/'))
        if '.DS_Store' in peo_pics:  # 去掉不为jpg和png格式的mac os系统文件
            peo_pics.remove('.DS_Store')
        for pic in peo_pics:
            peo_i += 1
            p_path = pics_path + '/' + peo + '/' + pic
            print(peo, peo_i, p_path)
            img_crop = cv2.imread(p_path)
            img_crop = np.asarray(cv2.resize(img_crop, image_size))
            images.append(img_crop)
            if name_is_folder == 1:
                fns.append(peo + '-' + str(peo_i))  # 文件夹名字+第几张照片
            else:
                fns.append(pic.split('.')[0])  # 照片原名字
    print('load pic done!', Counter(fns))
    return np.array(images), fns
