from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
# ! coding=utf-8
import json
import os
from flask import Flask, render_template, Response, redirect, request
import cv2
from imutils.video import VideoStream
import numpy as np
from align import mtcnn_def_trans as fc_server
import time
import pickle
from data_pro.data_utils import brenner
import multiprocessing
from data_pro.data_utils import is_birthday
# from rg_model.model_facenet import FacenetPre
# facenet_pre_m = FacenetPre()
# lastsave_embs0 = np.zeros(128)
# imgsize = 160
# trans_01 = 0
# facenet_pre_m.gen_knowns_db('../facenet_files/office_face160/', '../facenet_files/embs_pkl/facenet/')

# from rg_model.model_insight_lucky import InsightPreLucky
# facenet_pre_m = InsightPreLucky()
# lastsave_embs0 = np.zeros(512)
# imgsize = 112
# # facenet_pre_m.gen_knowns_db('../facenet_files/office_face160/', '../facenet_files/embs_pkl/insight_luck/')

from rg_model.model_insight_auroua import InsightPreAuroua

'''加载模型和已知人脸库'''
facenet_pre_m = InsightPreAuroua()
lastsave_embs0 = np.zeros(512)
imgsize = 112
trans_01 = 0
pic_path = '/Users/finup/Desktop/rg/face_rg_files/common_files/dc_marking_trans_newnanme/'
pkl_path = '../face_rg_files/embs_pkl/ep_insight_auroua/50w_dc_all.pkl'
# facenet_pre_m.gen_knowns_db(pic_path, pkl_path)

'''设置全局变量'''
# flask 类
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
# 多进程类
multi_p = None
# 摄像头类
camera, c_w, c_h = None, 1280, 720
# 读取所有员工信息
office_p = str(
    os.path.abspath(__file__).split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/office_info.pkl"
fr = open(office_p, 'rb')
all_officeinfo_dct = pickle.load(fr)  # {'046115':['孙瑞娜', 'sunruina', '19930423', []]}
print('all_office_num', len(all_officeinfo_dct))
# 实时识别结果存储，长度为帧中人数+1
frame_rg_list = [[],
                 {}]  # 无人：[img视频原图]，有人：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb} , ... , {工号n: 46119，人脸图片: crop_img，向量: emb} ]
# 上一帧结果临时存储
last_rg_list = [[],
                {}]  # 无人：[img视频原图]，有人：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb} , ... , {工号n: 46119，人脸图片: crop_img，向量: emb} ]
# 最新拍照信息的识别结果
photo_rg_list = [[], {}]  # 无效：[]，仅1人时有效变量长度为2：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb} ]
# 流程监控
monitor_dct = {'add_n': 0}
# 模型超参
para_dct = {'mtcnn_minsize': int(0.2 * min(c_w, c_h)), 'clear_day': 50, 'clear_night': 50,
            'savepic_path': '../face_rg_files/save_pics/stream/'}


# names = []
# faceembs = []
# f_areas_r_lst = []
# names1p_last_n = []
# realtime = True
# capture_saved = False
# new_photo = None
# save_flag = 0
# add_faces_n = 0
# camera = None
# c_w, c_h = 1280, 720
# # c_w, c_h = 1920, 1080
# mtcnn_minsize = int(0.2 * min(c_w, c_h))
# frame = None
# p = None
# status = ''
# take_photo_pic = None


def rg_1frame(f_pic):
    global para_dct, frame_rg_list, last_rg_list, photo_rg_list
    now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # 统计每步执行时间
    # last_exetime_rg = time.time()
    # now_exetime_rg = time.time()
    # print('TIME rg: ********************************************** Start', np.round((now_exetime_rg - last_exetime_rg), 4))
    # last_exetime_rg = now_exetime_rg

    dets, crop_images, point5, det_flag = fc_server.load_and_align_data(f_pic, trans_flag=trans_01, image_size=imgsize,
                                                                 minsize=para_dct[
                                                                     'mtcnn_minsize'])  # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片

    # now_exetime_rg = time.time()
    # print('TIME rg: aligin', np.round((now_exetime_rg - last_exetime_rg), 4))
    # last_exetime_rg = now_exetime_rg

    if det_flag != 0:
        # 清晰度过滤，仅检测人脸概率最大的那个人的清晰度。
        is_qingxi1, is_qingxi0 = brenner(cv2.cvtColor(crop_images[0], cv2.COLOR_BGR2GRAY))  # is_qingxi1是枞向运动模糊方差，0是横向
        now_hour = int(now_time[8:10])
        if now_hour >= 17:
            qx_hold = para_dct['clear_night']
        else:
            qx_hold = para_dct['clear_day']
        if is_qingxi1 >= qx_hold and is_qingxi0 >= qx_hold:  # 有人且清晰，则画人脸，进行识别名字

            names, faceis_konwns, faceembs, min_sims = facenet_pre_m.imgs_get_names(crop_images)  # 获取人脸名字
            names_cut = [i.split('-')[1] for i in names]
            ids_cut = [i.split('-')[0] for i in names]

            # 抽样存储识别图片,图片命名：时间戳 + 是否和上一张同人 + 库中最相似的相似度 + 大小 + 识别名字结果
            if len(names) == 1:
                sead = hash(str(time.time())[-6:])
                if sead % 2 == 1:  # hash采样
                    if len(last_rg_list) != 0:  # 有上一帧
                        is_same_p = facenet_pre_m.d_cos(faceembs[0], last_rg_list[1]['emb1'])
                        if is_same_p[0] > 0.85:
                            is_same_t = '1'
                        else:
                            is_same_t = '0'

                        fpic_path = para_dct['savepic_path'] + now_time + '_' + is_same_t + '-' + str(
                            is_same_p[0])[2:4] + '_' + str(int(is_qingxi1)) + '-' + str(int(is_qingxi0)) + '_' + str(
                            min_sims[0])[2:4]
                        cv2.imwrite(fpic_path + '_crop_@' + names[0] + '.jpg', crop_images[0])
                        cv2.imwrite(fpic_path + '_raw_@' + names[0] + '.jpg', f_pic)
                        last_rg_list = frame_rg_list  # 更新last save emb，以便判定本帧是否和上一帧同一个人

                    # 画鼻子眼睛保存，
                    # print('标记5点位置', point5[0])
                    # crop_img_mark = fc_server.mark_face_points(point5, crop_images[0])
                    # mark5 = ''
                    # for i in point5:
                    #     mark5 += str(int(i[0])) + '-'
                    # mark5 = mark5[0:-1]
                    # cv2.imwrite(fpic_path + '_cropmark' + '_' + mark5 + '_' + names[0] + '.jpg', crop_img_mark)

            # 绘制矩形框并标注文字
            # f_pic_new, f_areas_r_lst = fc_server.mark_pic(dets, names_cut, f_pic)

            # now_exetime_rg = time.time()
            # print('TIME rg: draw pic', np.round((now_exetime_rg - last_exetime_rg), 4))
            # last_exetime_rg = now_exetime_rg

            # #  对人脸名字进行稳定性修正
            # if len(names_cut) == 5:
            #     if len(names1p_last_n) <= 1:  # 不到最近5帧继续追加
            #         names1p_last_n.append(names_cut[0])
            #     else:  # 到5帧则去掉最早的，留下最近的
            #         del names1p_last_n[0]
            #         names1p_last_n.append(names_cut[0])
            #
            #     top_names = Counter(names1p_last_n).most_common(2)  # 依据最近5帧统计top名字
            #     if len(top_names) >= 2:  # 如果名字种类两个名字以上
            #         name_top1 = top_names[0][0]
            #         name_top2 = top_names[1][0]
            #         if name_top1 == '未知的同学':  # 如果名字种类两个名字以上，且第一名是未知，则返回第二个，即已知
            #             names_cut[0] = name_top2
            #         else:  # 如果名字种类两个名字以上，且Top1是已知，则返回Top1
            #             names_cut[0] = name_top1

            # return f_pic_new
        else:  # 有人但概率最大的那张人脸不清晰，则只画人脸
            ids_cut = ['不清晰' for i in dets]
            faceembs = [[] for i in dets]
    else:  # 没有人
        ids_cut = ['无人' for i in dets]
        faceembs = [[] for i in dets]

    # frame_rg_list 无人：[img视频原图]，有人：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb}, ...]
    frame_rg_list = [f_pic]
    if len(dets) != 0:
        for p_i in range(len(dets)):
            p1_rg_res = {'p1_id': ids_cut[p_i], 'p1_crop': crop_images[p_i], 'p1_emb': faceembs[p_i]}
            frame_rg_list.append(p1_rg_res)
    else:  # 无人
        frame_rg_list.append({'p1_id': '无人', 'p1_crop': [], 'p1_emb': []})
    return frame_rg_list


def camera_open():
    camera = VideoStream(0)
    camera.stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    camera.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, c_w)
    camera.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, c_h)
    camera.start()
    return camera


def f():
    global camera
    if not camera:
        camera = camera_open()

    i = 0  # 统计fps的时间
    start_flag = time.time()
    while 1:
        frame = camera.read()

        i += 1  # 统计fps的时间,计算每间隔了1s，会处理几张frame
        interval = int(time.time() - start_flag)
        if interval == 1:
            print('#########################################################fps:', i, '  add_n:', monitor_dct['add_n'])
            start_flag = time.time()
            i = 0

        # # 每天23点00分的第一帧的时间下进行一次name_embs存储，因为可能有重名的问题，所以不能存为dict
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        # if time_stamp[8:13] == ('2300' + '0'):
        #     save_flag += 1
        #     if save_flag == 1:
        #         with open("../facenet_files/pickle_files/" + time_stamp + "_names_lst.pkl", 'wb') as f1:
        #             pickle.dump(facenet_pre_m.known_names, f1)
        #         with open("../facenet_files/pickle_files/" + time_stamp + "_embs_lst.pkl", 'wb') as f2:
        #             pickle.dump(facenet_pre_m.known_names, f2)
        # if time_stamp[8:13] == ('2259' + '0'):
        #     save_flag = 0

        if frame is not None:
            frame = cv2.flip(frame, 1)  # 前端输出镜面图片
            rg_1frame(frame)

        else:
            # print('img_None')
            camera = camera_open  # 摄像头失效抓取为空，则重启摄像头
            continue


@app.route('/')
def index():
    return render_template('index_ttt.html')


@socketio.on('connect', namespace='/test_conn')  # 建立通讯连接
def test_message():
    print('client connect to test_conn')
    global multi_p
    multi_p = multiprocessing.Process(target=f)
    multi_p.start()


@socketio.on('get_name', namespace='/test_conn')  # 消息实时传送
def test_message(message):
    # print(message)
    print('111111111111111111111111')
    global frame_rg_list, all_officeinfo_dct
    res_json = {'app_data': {}, 'app_status': '1'}

    p1_id = frame_rg_list[1]['p1_id']
    if p1_id not in ['不清晰', '无人']:

        for i in range(len(frame_rg_list)):
            if i == 0:  # list里第一项是原图，此处不需要，photo的时候才需要
                pass
            elif i in [1, 2, 3, 4]:
                c_name = all_officeinfo_dct[p1_id][0]
                e_name = all_officeinfo_dct[p1_id][1]
                is_birth = is_birthday(all_officeinfo_dct[p1_id][2])
                _, jpeg = cv2.imencode('.jpg', frame_rg_list[i]['p1_crop'])
                crop_img = jpeg.tobytes()

                res_json['rg_data']['P_' + str(i)] = {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name,
                                                      'is_birth': is_birth, 'crop_img': crop_img}
            else:  # 只显示前人脸概率最大的前4个人
                break
    else:
        res_json = {'app_data': {}, 'app_status': '0'}

    # res_json = {'rg_data': {'工号0': {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name, 'is_birth': is_birth, 'crop_img': crop_img},
    #                         '工号1': {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name, 'is_birth': is_birth, 'crop_img': crop_img}},
    #             'rg_status': '1'}

    res_json = json.dumps(res_json)
    emit('get_video_response', res_json)


@socketio.on('get_video', namespace='/test_conn')  # 消息实时传送
def test_message(message):
    # print(message)
    global frame_rg_list
    res_json = {'app_data': {}, 'app_status': '1'}

    raw_pic = frame_rg_list[0]
    _, raw_pic = cv2.imencode('.jpg', raw_pic)
    res_json['app_data']['video_pic'] = raw_pic.tobytes()

    res_json = json.dumps(res_json)
    emit('get_video_response', res_json)


@socketio.on('lock_video', namespace='/test_conn')  # 消息实时传送
def test_message(message):
    global frame_rg_list, photo_rg_list
    res_json = {'app_data': {}, 'app_status': '1'}

    if len(frame_rg_list) == 2:
        raw_pic = frame_rg_list[0]
        _, raw_pic = cv2.imencode('.jpg', raw_pic)
        res_json['app_data']['video_pic'] = raw_pic.tobytes()
        photo_rg_list = frame_rg_list
    else:
        photo_rg_list = []

    res_json = json.dumps(res_json)
    emit('get_video_response', res_json)


@socketio.on('add_new', namespace='/test_conn')  # 消息实时传送
def test_message(message):
    global frame_rg_list, photo_rg_list, monitor_dct
    para = dict(message)
    p_id = para['P1']
    p_angle = para['P2']
    res_json = {'app_data': {}, 'app_status': '1'}
    # 只差这里了

    if p_id in all_officeinfo_dct.keys():

        if len(photo_rg_list) == 2:
            raw_pic = photo_rg_list[0]
            p1_id = photo_rg_list[1]['p1_id']
            p1_crop = photo_rg_list[1]['p1_crop']
            p1_emb = photo_rg_list[1]['p1_emb']
            time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            pic_name = p1_id + '-' + all_officeinfo_dct[p1_id]['c_name'] + '-' + time_stamp
            monitor_dct['add_n'] += 1
            facenet_pre_m.known_embs = np.insert(facenet_pre_m.known_embs, 0, values=np.asarray(p1_emb), axis=0)
            facenet_pre_m.known_vms = np.insert(facenet_pre_m.known_vms, 0, values=np.linalg.norm(p1_emb), axis=0)
            facenet_pre_m.known_names = np.insert(facenet_pre_m.known_names, 0, values=np.asarray(pic_name), axis=0)
            cv2.imwrite(para_dct['savepic_path'] + 'photos/' + pic_name + '_crop.jpg', raw_pic)
            cv2.imwrite(para_dct['savepic_path'] + 'photos/' + pic_name + '_raw.jpg', raw_pic)

            res_json['app_data'] = {'message': '录入成功'}
        else:
            res_json['app_status'] = '2'
            res_json['app_data'] = {'message': '照片无效'}
    else:
        res_json['app_status'] = '0'
        res_json['app_data'] = {'message': '工号不存在'}

    res_json = json.dumps(res_json)
    emit('get_video_response', res_json)


if __name__ == '__main__':
    # socketio.run(app, debug=True, port=5000)
    socketio.run(app, port=5000)
