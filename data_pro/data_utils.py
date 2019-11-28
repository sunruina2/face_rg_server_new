import os
import cv2
import numpy as np
from collections import Counter


def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i + batch_size, data_num), ...]


def load_image(pics_path, image_size, name_is_folder=0, print_Counter=1):
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
                fns.append(pic.split('.')[0])  # 取照片原名字
    if print_Counter == 1:
        print('load pic done!', Counter([i.split('-')[1] for i in fns]))
    return np.array(images), fns


def brenner(img_i):
    img_i = np.asarray(img_i, dtype='float64')
    x, y = img_i.shape
    img_i -= 127.5
    img_i *= 0.0078125  # 标准化
    center = img_i[0:x - 2, 0:y - 2]
    center_xplus = img_i[2:, 0:y - 2]
    center_yplus = img_i[0:x - 2:, 2:]
    Dx = np.sum((center_xplus - center) ** 2)
    Dy = np.sum((center_yplus - center) ** 2)
    return Dx, Dy


def is_birthday(birthday):
    # birthday '19991112'
    import time
    today = time.localtime()
    birthday = time.strptime(birthday, "%Y%m%d")
    if birthday[1] == today[1] and birthday[2] == today[2]:
        return '1'
    else:
        return '0'


def del_side(p5, img_size):
    # _landmark = np.asarray(
    #     [[['1左眼x', '1左眼y'],
    #       ['1右眼x', '1右眼y'],
    #       ['1鼻子x', '1鼻子y'],
    #       ['1左嘴x', '1左嘴y'],
    #       ['1右嘴x', '1右嘴y'], ],
    #      [['2左眼x', '2左眼y'],
    #       ['2右眼x', '2右眼y'],
    #       ['2鼻子x', '2鼻子y'],
    #       ['2左嘴x', '2左嘴y'],
    #       ['2右嘴x', '2右嘴y'], ],
    #      [['3左眼x', '3左眼y'],
    #       ['3右眼x', '3右眼y'],
    #       ['3鼻子x', '3鼻子y'],
    #       ['3左嘴x', '3左嘴y'],
    #       ['3右嘴x', '3右嘴y'], ]]
    front_index = []
    for i in range(len(p5)):
        if p5[i, 0, 0] < img_size * 0.33:  # 左眼x在左1/3范围内
            if p5[i, 1, 0] > img_size * 0.66:  # 右眼x在右1/3范围内
                if img_size * 0.5 < p5[i, 2, 1] < img_size * 0.85:  # 鼻尖y在crop图片下0.5 -0.85范围内
                    front_index.append(i)

    return front_index


def info_dct(info_p, save_p):
    import pandas as pd
    import pickle
    o_info = np.asarray(pd.read_csv(info_p, dtype='str'), dtype='str')
    info_dict = {}
    for i in range(len(o_info)):
        if o_info[i][2] == 'nan':
            o_info[i][2] = ''
        info_dict[o_info[i][0]] = [o_info[i][1], o_info[i][2], o_info[i][3], []]

    if len(info_dict) != 0:
        # 存已知人脸embs dict
        with open(save_p, 'wb') as f:
            pickle.dump(info_dict, f)
        print('saving office pkl...', len(info_dict), save_p)


def name2idname(pics_path, save_path):
    import shutil
    import pandas as pd

    from string import digits

    exe_path = os.path.abspath(__file__)
    f_path = str(exe_path.split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/office_info.csv"
    o_info = np.asarray(pd.read_csv(f_path, dtype='str'), dtype='str')
    print(o_info.shape[0])
    name_id_dct = dict(zip(list(o_info[:, 1]), list(o_info[:, 0])))
    print('共计有重名人数：', o_info.shape[0] - len(name_id_dct))

    try:
        os.mkdir(save_path)
    except:
        pass

    all_knowembs_list = []
    print('pic reading %s' % pics_path)
    if os.path.isdir(pics_path):
        paths = list(os.listdir(pics_path))
        if '.DS_Store' in paths:  # 去掉不为文件夹格式的mac os系统文件
            paths.remove('.DS_Store')
    else:
        paths = [pics_path]
    for peo in paths:
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', peo)
        peo1_i = 0
        peo_pics = list(os.listdir(pics_path + '/' + peo + '/'))

        if peo == '王海峰':
            peo_new = '王海锋'
        elif peo == '陈寅':
            continue
        elif peo == '曹元青':
            continue
        else:
            peo_new = peo

        if peo == '李平':
            p_id = '049577'
        elif peo == '王鹏':
            p_id = '018576'
        elif peo == '张伟':
            p_id = '047259'
        elif peo == '李欣':
            p_id = '040582'
        elif peo == '付婷婷':
            p_id = '027474'
        elif peo == '张颖':
            p_id = '001403'
        elif peo == '王利':
            p_id = '053009'
        elif peo == '马超':
            p_id = '003583'
        elif peo == '张林':
            p_id = '046112'
        elif peo == '杨帆':
            p_id = '000004'
        else:
            p_id = name_id_dct.get(peo_new)

        if '.DS_Store' in peo_pics:  # 去掉不为jpg和png格式的mac os系统文件
            peo_pics.remove('.DS_Store')
        try:
            os.mkdir(save_path + '/' + p_id + '-' + peo_new + '/')
        except:
            pass

        for pic in peo_pics:
            peo1_i += 1
            p_path = pics_path + '/' + peo + '/' + pic
            id = name_id_dct.get(peo)
            print(peo, id)
            new_p_path = s_path + '/' + p_id + '-' + peo_new + '/' + p_id + '-' + peo_new + '-' + str(peo1_i) + '.jpg'
            srcfile = p_path
            dstfile = new_p_path
            print(srcfile)
            print(dstfile)
            shutil.copyfile(srcfile, dstfile)

    # print('load pic done!', Counter())
    #
    # import pickle
    # if len(all_knowembs_list) != 0:
    #     # 存已知人脸embs dict
    #     with open(save_p, 'wb') as f:
    #         pickle.dump(all_knowembs_list, f)
    #     print('saving office pkl...', len(all_knowembs_list), save_p)

    def office_crop():
        office_crop_dict = {}
        import re
        pic_name = '孙瑞娜那046115数据中心'
        crop_pic = cv2.imread(pic_name)
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, pic_name)

        for pid in res:
            if len(pid) == 6:
                office_crop_dict[pid] = crop_pic
            else:
                pass


if __name__ == '__main__':
    '''生成所有员工info信息'''
    # exe_path = os.path.abspath(__file__)
    # f_path = str(exe_path.split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/office_info.csv"
    # s_path = str(exe_path.split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/office_info.pkl"
    # info_dct(f_path, s_path)

    '''已知员工照片目录nameN改为员工号-name-N'''
    # exe_path = os.path.abspath(__file__)
    # pic_path = str(exe_path.split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/dc_marking_trans"
    # s_path = str(exe_path.split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/dc_marking_trans_newnanme"
    # print(pic_path, s_path)
    # name2idname(pic_path, s_path)
