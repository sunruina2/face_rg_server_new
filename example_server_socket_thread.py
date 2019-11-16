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
from anti.anti_pre import AntiSpoofing
from rg_model.model_insight_auroua import InsightPreAuroua
from threading import Lock

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

'''加载模型和已知人脸库'''
facenet_pre_m = InsightPreAuroua()
lastsave_embs0 = np.zeros(512)
imgsize = 112
trans_01 = 0
pic_path = '/Users/finup/Desktop/rg/face_rg_files/common_files/dc_marking_trans_newnanme/'
pkl_path = '../face_rg_files/embs_pkl/ep_insight_auroua/50w_dc_all.pkl'
# facenet_pre_m.gen_knowns_db(pic_path, pkl_path)

'''加在活体检测模型'''
# anti = AntiSpoofing()

'''读取所有员工信息'''
office_p = str(
    os.path.abspath(__file__).split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/office_info.pkl"
fr = open(office_p, 'rb')
all_officeinfo_dct = pickle.load(fr)  # {'046115':['孙瑞娜', 'sunruina', '19930423', []]}
print('all_office_num', len(all_officeinfo_dct))

'''设置全局变量'''
# flask 类
async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()
# 多进程类
multi_p = None
# 摄像头类
camera, c_w, c_h = None, 1280, 720
# 实时识别结果存储，长度为帧中人数+1
frame_rg_list = [[], {}]
# 无人：[img视频原图]，有人：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb} , ... , {工号n: 46119，人脸图片: crop_img，向量: emb} ]
# 单人模式上一帧emb临时存储，当前人的frame流历史最大准确识别
last_1p_emb = np.zeros(512)
his_maxacc = {'max_name': '', 'max_sim': 0.0}
# 最新拍照信息的识别结果
photo_rg_list = [[], {}]  # 无效：[]，仅1人时有效变量长度为2：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb} ]
# 流程监控
monitor_dct = {'add_n': 0}
# 模型超参
para_dct = {'mtcnn_minsize': int(0.2 * min(c_w, c_h)), 'night_start_h': 17, 'clear_day': 100, 'clear_night': 50,
            'savepic_path': '../face_rg_files/save_pics/stream/',
            'rg_sim': 0.75, 'frame_sim': 0.8, 'det_para': [256, 0.1, 112, 0.0]}


def rg_1frame(f_pic):
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:rg_1frame')

    global para_dct, frame_rg_list, last_1p_emb, his_maxacc
    now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片
    dets, crop_images_at, point5_at, crop_images_rg, point5_rg, align_flag = fc_server.load_and_align_data(f_pic,
                                                                                                           para_dct[
                                                                                                               'det_para'],
                                                                                                           minsize=90)

    if align_flag != 0:
        # 清晰度过滤，仅检测人脸概率最大的那个人的清晰度。
        is_qingxi1, is_qingxi0 = brenner(
            cv2.cvtColor(np.asarray(crop_images_rg[0], np.float32), cv2.COLOR_BGR2GRAY))  # is_qingxi1是枞向运动模糊方差，0是横向
        now_hour = int(now_time[8:10])
        if now_hour >= para_dct['night_start_h']:
            qx_hold = para_dct['clear_night']
        else:
            qx_hold = para_dct['clear_day']
        if is_qingxi1 >= qx_hold and is_qingxi0 >= qx_hold:  # 有人且清晰，则画人脸，进行识别名字

            '''人脸识别'''
            names, faceis_konwns, faceembs, sims = facenet_pre_m.imgs_get_names(crop_images_rg,
                                                                                batchsize=len(dets))  # 获取人脸名字
            ids_cut = [i.split('-')[0] for i in names]
            '''活体检测'''
            # anti_flag_list = []
            # for i in range(len(dets)):
            #     anti_flag_list.append(anti(crop_images_rg[i]))
            # '''在画框上显示是否活体'''
            # for i in range(len(dets)):
            #     names[i] = names[i] + 'at' + str(anti_flag_list[i])

            # 抽样存储识别图片,图片命名：时间戳 + 是否和上一张同人 + 库中最相似的相似度 + 大小 + 识别名字结果
            if len(names) == 1:

                is_same_p = facenet_pre_m.d_cos(faceembs[0], last_1p_emb)
                if is_same_p[0] > para_dct['frame_sim']:
                    is_same_t = '1'
                else:
                    is_same_t = '0'

                if is_same_t == '1':
                    if sims[0] > his_maxacc['max_sim']:
                        # 如果遇到这个人更准确的脸部识别结果则用该最准结果作为输出结果。并把此刻最终结果存储在 历史最准字典中
                        his_maxacc = {'max_name': names[0], 'max_sim': sims[0]}
                        # names[0] = names[0] + '_rg'+names[0]
                    else:
                        # 否则用这个人过去实时流中识别最准的识别结果。
                        names[0] = his_maxacc['max_name']
                        # names[0] = his_maxacc['max_name'] + '_rg'+names[0]
                else:
                    # 如果换人了，则将单人历史最准记录置零，再开始新的人的从0更新
                    his_maxacc = {'max_name': '', 'max_sim': 0.0}

                # 单人模式实时流随机保存
                sead = hash(str(time.time())[-6:])
                if sead % 2 == 1:  # hash采样
                    fpic_path = para_dct['savepic_path'] + now_time + '_' + is_same_t + '-' + str(
                        is_same_p[0])[2:4] + '_' + str(int(is_qingxi1)) + '-' + str(int(is_qingxi0)) + '_' + str(
                        sims[0])[2:4]
                    cv2.imwrite(fpic_path + '_crop_@' + names[0] + '.jpg', crop_images_rg[0])
                    cv2.imwrite(fpic_path + '_raw_@' + names[0] + '.jpg', f_pic)
                last_1p_emb = faceembs[0]  # 更新last save emb，以便判定本帧是否和上一帧同一个人
            else:  # 多人，因为检测是识别两个过程，无法对应是否同人，所有不能作任何识别结果修正，直接返回模型识别结果即可
                pass
        else:  # 有人但概率最大的那张人脸不清晰，则不管单人多人，全都返回不清晰
            ids_cut = ['不清晰' for i in dets]
            faceembs = [[] for i in dets]
    else:  # 没有人
        ids_cut = ['无人' for i in dets]
        faceembs = [[] for i in dets]

    # 构造frame_rg_list 无人：[img视频原图]，有人：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb}, ...]
    frame_rg_list = [f_pic]
    if len(dets) != 0:
        for p_i in range(len(dets)):
            p1_rg_res = {'p1_id': ids_cut[p_i], 'p1_crop': crop_images_rg[p_i], 'p1_emb': faceembs[p_i]}
            frame_rg_list.append(p1_rg_res)
    else:  # 无人
        frame_rg_list.append({'p1_id': '无人', 'p1_crop': [], 'p1_emb': []})

    # 离线测试
    # get_name_message_test()
    # get_video_message_test()
    # lock_video_message_test()
    # add_new_message_test({'P1': '04611511', 'P2': '正面'})

    return frame_rg_list


def camera_open():
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:camera_open')
    camera1 = VideoStream(0)
    camera1.stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    camera1.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, c_w)
    camera1.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, c_h)
    camera1.start()
    return camera1


def background_thread():
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:f')

    global camera
    if not camera:
        camera = camera_open()
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:f camera')

    i = 0  # 统计fps的时间
    start_flag = time.time()
    while 1:
        frame = camera.read()
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:f while 1')

        i += 1  # 统计fps的时间,计算每间隔了1s，会处理几张frame
        interval = int(time.time() - start_flag)
        if interval == 1:
            print('#########################################################fps:', i, '  add_n:', monitor_dct['add_n'])
            start_flag = time.time()
            i = 0

        if frame is not None:
            frame = cv2.flip(frame, 1)  # 前端输出镜面图片
            rg_1frame(frame)

        else:
            # print('img_None')
            camera = camera_open  # 摄像头失效抓取为空，则重启摄像头
            continue


@app.route('/')
def index():
    return render_template('index_thread.html', async_mode=socketio.async_mode)


@socketio.on('connect', namespace='/test_conn')
def connect_message():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

    res_json = {'app_status': '1', 'app_data': {'message': '链接成功'}}
    res_json = json.dumps(res_json)
    emit('connect_response', res_json)


@socketio.on('get_name', namespace='/test_conn')  # 消息实时传送
def get_name_message(message):
    global frame_rg_list, all_officeinfo_dct
    res_json = {'app_data': {'message': '识别成功'}, 'app_status': '1'}
    p1_id = frame_rg_list[1]['p1_id']
    if p1_id not in ['不清晰', '无人']:

        for i in range(len(frame_rg_list)):
            if i == 0:  # list里第一项是原图，此处不需要，take photo的时候才需要
                pass
            elif i in [1, 2, 3, 4]:
                if all_officeinfo_dct[p1_id][0] != '0':
                    c_name = all_officeinfo_dct[p1_id][0]
                    e_name = all_officeinfo_dct[p1_id][1]
                    is_birth = is_birthday(all_officeinfo_dct[p1_id][2])
                else:
                    c_name = '未识别的同学'
                    e_name = 'unknown'
                    is_birth = '0'
                # _, jpeg = cv2.imencode('.jpg', frame_rg_list[i]['p1_crop'])
                # crop_img = jpeg.tobytes()

                crop_img = frame_rg_list[i]['p1_crop'].tolist()

                res_json['app_data']['P_' + str(i)] = {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name,
                                                       'is_birth': is_birth, 'crop_img': crop_img}
            else:  # 只显示前人脸概率最大的前4个人
                break
    else:
        res_json = {'app_data': {'message': '本帧无效'}, 'app_status': '1'}

    # res_json = {'app_data': {'message': '识别成功'
    #                         '工号0': {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name, 'is_birth': is_birth, 'crop_img': crop_img},
    #                         '工号1': {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name, 'is_birth': is_birth, 'crop_img': crop_img}},
    #             'app_status': '1'}

    res_json = json.dumps(res_json)
    # res_json = {'app_data': {'message': '识别成功', 'P_1': {'p1_id': '046115', 'c_name': '孙瑞娜', 'e_name': 'sunruina', 'is_birth': '0',
    #                                          'crop_img': [[[8.0, 9.0, 11.0], [121.0, 134.0, 152.0]]]}}, 'app_status': '1'}
    emit('get_video_response', res_json)


@socketio.on('get_video', namespace='/test_conn')  # 消息实时传送
def get_video_message(message):
    global frame_rg_list
    res_json = {'app_data': {'message': '获取实时帧成功'}, 'app_status': '1'}

    # raw_pic = frame_rg_list[0]
    # _, raw_pic = cv2.imencode('.jpg', raw_pic)
    # res_json['app_data']['video_pic'] = raw_pic.tobytes()

    res_json['app_data']['video_pic'] = frame_rg_list[0].tolist()

    res_json = json.dumps(res_json)
    # res_json = {"app_data": {"message": "获取实时帧成功", "video_pic": []}, "app_status": "1"}
    emit('get_video_response', res_json)


@socketio.on('lock_video', namespace='/test_conn')  # 消息实时传送
def lock_video_message(message):
    global frame_rg_list, photo_rg_list
    res_json = {'app_data': {'message': '图片有效'}, 'app_status': '1'}

    if len(frame_rg_list) == 2 and frame_rg_list[1]['p1_id'] not in ['无人', '不清晰']:
        # raw_pic = frame_rg_list[0]
        # _, raw_pic = cv2.imencode('.jpg', raw_pic)
        # res_json['app_data']['video_pic'] = raw_pic.tobytes()
        res_json['app_data']['video_pic'] = frame_rg_list[0].tolist()
        res_json['app_data']['video_pic'] = []

        photo_rg_list = frame_rg_list

    else:
        res_json = {'app_data': {'message': '图片无效'}, 'app_status': '0'}
        photo_rg_list = []
    res_json = json.dumps(res_json)
    emit('get_video_response', res_json)


@socketio.on('add_new', namespace='/test_conn')  # 消息实时传送
def add_new_message(message):
    global frame_rg_list, photo_rg_list, monitor_dct
    para = dict(message)
    p_id_input = para['P1']
    p_angle_input = para['P2']
    res_json = {'app_data': {}, 'app_status': '1'}
    # 只差这里了

    if p_id_input in all_officeinfo_dct.keys():

        if len(photo_rg_list) == 2:
            raw_pic = photo_rg_list[0]
            p1_crop = photo_rg_list[1]['p1_crop']
            p1_emb = photo_rg_list[1]['p1_emb']
            time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            pic_name = p_id_input + '-' + all_officeinfo_dct[p_id_input][0] + '-' + time_stamp + p_angle_input
            monitor_dct['add_n'] += 1

            facenet_pre_m.known_embs = np.insert(facenet_pre_m.known_embs, 0, values=np.asarray(p1_emb), axis=0)
            facenet_pre_m.known_vms = np.insert(facenet_pre_m.known_vms, 0, values=np.linalg.norm(p1_emb), axis=0)
            facenet_pre_m.known_names = np.insert(facenet_pre_m.known_names, 0, values=np.asarray(pic_name), axis=0)
            cv2.imwrite(para_dct['savepic_path'] + 'photos/' + pic_name + '_crop.jpg', p1_crop)
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


def get_name_message_test():
    global frame_rg_list, all_officeinfo_dct
    res_json = {'app_data': {'message': '识别成功'}, 'app_status': '1'}
    p1_id = frame_rg_list[1]['p1_id']
    if p1_id not in ['不清晰', '无人']:

        for i in range(len(frame_rg_list)):
            if i == 0:  # list里第一项是原图，此处不需要，take photo的时候才需要
                pass
            elif i in [1, 2, 3, 4]:
                if all_officeinfo_dct[p1_id][0] != '0':
                    c_name = all_officeinfo_dct[p1_id][0]
                    e_name = all_officeinfo_dct[p1_id][1]
                    is_birth = is_birthday(all_officeinfo_dct[p1_id][2])
                else:
                    c_name = '未识别的同学'
                    e_name = 'unknown'
                    is_birth = '0'
                # _, jpeg = cv2.imencode('.jpg', frame_rg_list[i]['p1_crop'])
                # crop_img = jpeg.tobytes()

                crop_img = frame_rg_list[i]['p1_crop'].tolist()

                res_json['app_data']['P_' + str(i)] = {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name,
                                                       'is_birth': is_birth, 'crop_img': crop_img}
            else:  # 只显示前人脸概率最大的前4个人
                break
    else:
        res_json = {'app_data': {'message': '本帧无效'}, 'app_status': '1'}

    # res_json = {'app_data': {'message': '识别成功'
    #                         '工号0': {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name, 'is_birth': is_birth, 'crop_img': crop_img},
    #                         '工号1': {'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name, 'is_birth': is_birth, 'crop_img': crop_img}},
    #             'app_status': '1'}
    if len(res_json['app_data'].keys()) > 2:
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&', res_json)
    res_json = json.dumps(res_json)


def get_video_message_test():
    global frame_rg_list
    res_json = {'app_data': {'message': '获取实时帧成功'}, 'app_status': '1'}

    # raw_pic = frame_rg_list[0]
    # _, raw_pic = cv2.imencode('.jpg', raw_pic)
    # res_json['app_data']['video_pic'] = raw_pic.tobytes()

    res_json['app_data']['video_pic'] = frame_rg_list[0].tolist()

    res_json = json.dumps(res_json)
    # res_json = {"app_data": {"message": "获取实时帧成功", "video_pic": []}, "app_status": "1"}

    print('rawrawrawrawrawrawrawrawraw', res_json)


def lock_video_message_test():
    global frame_rg_list, photo_rg_list
    res_json = {'app_data': {'message': '图片有效'}, 'app_status': '1'}

    if len(frame_rg_list) == 2 and frame_rg_list[1]['p1_id'] not in ['无人', '不清晰']:
        # raw_pic = frame_rg_list[0]
        # _, raw_pic = cv2.imencode('.jpg', raw_pic)
        # res_json['app_data']['video_pic'] = raw_pic.tobytes()
        res_json['app_data']['video_pic'] = frame_rg_list[0].tolist()
        res_json['app_data']['video_pic'] = []

        photo_rg_list = frame_rg_list

    else:
        res_json = {'app_data': {'message': '图片无效'}, 'app_status': '0'}
        photo_rg_list = []
    res_json = json.dumps(res_json)


def add_new_message_test(message):
    global frame_rg_list, photo_rg_list, monitor_dct
    para = dict(message)
    p_id_input = para['P1']
    p_angle_input = para['P2']
    res_json = {'app_data': {}, 'app_status': '1'}
    # 只差这里了

    if p_id_input in all_officeinfo_dct.keys():

        if len(photo_rg_list) == 2:
            raw_pic = photo_rg_list[0]
            p1_crop = photo_rg_list[1]['p1_crop']
            p1_emb = photo_rg_list[1]['p1_emb']
            time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            pic_name = p_id_input + '-' + all_officeinfo_dct[p_id_input][0] + '-' + time_stamp + p_angle_input
            monitor_dct['add_n'] += 1

            facenet_pre_m.known_embs = np.insert(facenet_pre_m.known_embs, 0, values=np.asarray(p1_emb), axis=0)
            facenet_pre_m.known_vms = np.insert(facenet_pre_m.known_vms, 0, values=np.linalg.norm(p1_emb), axis=0)
            facenet_pre_m.known_names = np.insert(facenet_pre_m.known_names, 0, values=np.asarray(pic_name), axis=0)
            cv2.imwrite(para_dct['savepic_path'] + 'photos/' + pic_name + '_crop.jpg', p1_crop)
            cv2.imwrite(para_dct['savepic_path'] + 'photos/' + pic_name + '_raw.jpg', raw_pic)
            res_json['app_data'] = {'message': '录入成功'}
        else:
            res_json['app_status'] = '2'
            res_json['app_data'] = {'message': '照片无效'}
    else:
        res_json['app_status'] = '0'
        res_json['app_data'] = {'message': '工号不存在'}

    res_json = json.dumps(res_json)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$', res_json)
    # emit('get_video_response', res_json)


if __name__ == '__main__':
    # socketio.run(app, debug=True, port=5000)
    socketio.run(app, port=5000)

    # test_message_def()
