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
import multiprocessing
from data_pro.data_utils import brenner, is_birthday, del_side
from anti.anti_pre import AntiSpoofing
from rg_model.model_insight_auroua import InsightPreAuroua
from threading import Lock
import sys

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
emb_pkl_path = '../face_rg_files/embs_pkl/ep_insight_auroua/'
'''生成新的人脸库'''
# pic_path = '../face_rg_files/common_files/dc_marking_all_trans/'
# facenet_pre_m.gen_knowns_db(pic_path, emb_pkl_path+'50w_dc_all.pkl')

'''加在活体检测模型'''
# anti = AntiSpoofing()

'''读取所有员工信息'''
office_p = str(
    os.path.abspath(__file__).split('face_rg_server_new/')[0]) + 'face_rg_files/' + "common_files/office_info.pkl"
fr = open(office_p, 'rb')
all_officeinfo_dct = pickle.load(fr)  # {'046115':['孙瑞娜', 'sunruina', '19930423', []]}
today_dt = str(time.localtime()[0]) + str(time.localtime()[1]) + str(time.localtime()[2])
all_officeinfo_dct['053009'][2] = today_dt
print('@@@: 改测试生日', all_officeinfo_dct['053009'])
print('@@@: All_office_IDn:', len(all_officeinfo_dct))

'''设置全局变量'''
# 接口调用状态
api_status = '静默'
# flask 类
async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins='*', binary=True)
thread = None
thread_lock = Lock()
# 多进程类
multi_p = None
# 摄像头类
camera, c_w, c_h = None, 1280, 720
# 实时识别结果存储，长度为帧中人数+1
frame_rg_list = [[], {'p1_id': '无人', 'p1_face': [], 'p1_crop': [], 'p1_emb': []}]
# 无人：[img视频原图]，有人：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb} , ... , {工号n: 46119，人脸图片: crop_img，向量: emb} ]
# 单人模式上一帧emb临时存储，当前人的frame流历史最大准确识别
last_1p_emb = np.zeros(512)
his_maxacc = {'max_name': '0-历史最准名默认值-0', 'max_sim': 0.0}
# 最新拍照信息的识别结果
photo_rg_list = [[], {'p1_id': '无人', 'p1_face': [], 'p1_crop': [], 'p1_emb': []}]
# 无效：[]，仅1人时有效变量长度为2：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb} ]
# 流程监控
monitor_dct = {'add_n_all': 0, 'add_n_saved': 0}
# 模型超参
para_dct = {'mtcnn_minsize': int(0.2 * min(c_w, c_h)), 'night_start_h': 17, 'clear_day': 100, 'clear_night': 50,
            'savepic_path': '../face_rg_files/save_pics/',
            'rg_sim': 0.8, 'frame_sim': 0.77, 'det_para': [112, 0.3, 112, 0.0],
            'video_size_r': 0.7}  # det_para = [at,at扩充r,rg,rg扩充r]
facenet_pre_m.rg_hold = para_dct['rg_sim']


def rg_1frame(f_pic):

    global para_dct, frame_rg_list, last_1p_emb, his_maxacc
    now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    now_hour = int(now_time[8:10])
    if now_hour >= para_dct['night_start_h']:
        qx_hold = para_dct['clear_night']
    else:
        qx_hold = para_dct['clear_day']

    # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片
    dets, crop_images_at, point5_at, crop_images_rg, point5_rg, align_flag = fc_server.load_and_align_data(f_pic,
                                                                                                           para_dct[
                                                                                                               'det_para'],
                                                                                                           minsize=90)
    # # 把外极度侧脸和低头仰头过滤掉
    # ok_index = del_side(point5_rg, para_dct['det_para'][2])
    # dets, crop_images_at, point5_at, crop_images_rg, point5_rg = dets[ok_index], crop_images_at[ok_index], point5_at[
    #     ok_index], crop_images_rg[ok_index], point5_rg[ok_index]
    # if len(dets) == 0:
    #     align_flag = 0
    if align_flag != 0:
        # 清晰度过滤，仅检测人脸概率最大的那个人的清晰度。is_qingxi1是枞向运动模糊方差，0是横向

        '''活体检测'''
        # anti_flag_list = []
        # for i in range(len(dets)):
        #     anti_flag_list.append(anti(crop_images_rg[i]))
        # '''在画框上显示是否活体'''
        # for i in range(len(dets)):
        #     names[i] = names[i] + 'at' + str(anti_flag_list[i])
        # 抽样存储识别图片,图片命名：时间戳 + 是否和上一张同人 + 库中最相似的相似度 + 大小 + 识别名字结果
        if len(dets) == 1:
            '''人脸识别'''
            # 1计算本帧是谁
            names, faceis_konwns, faceembs, sims = facenet_pre_m.imgs_get_names(crop_images_rg, batchsize=len(dets))
            # 2计算本帧是否同人
            is_same_p = facenet_pre_m.d_cos(faceembs[0], last_1p_emb)
            if is_same_p[0] > para_dct['frame_sim']:
                is_same_t = '1'
            else:
                is_same_t = '0'
            # 3计算本帧清晰度
            is_qx1, is_qx0 = brenner(cv2.cvtColor(np.asarray(crop_images_rg[0], np.float32), cv2.COLOR_BGR2GRAY))

            # 逻辑判断，先判断是否同一个人，再判断本帧清晰与否，不管清洗不清晰都要有返回结果，避免闪屏
            if is_same_t == '1':  # 同人
                if is_qx1 >= qx_hold and is_qx0 >= qx_hold:  # 同人，清晰，可用于更新结果变量his_maxacc，更新后取历史最高概率结果
                        if sims[0] > his_maxacc['max_sim']:  # 本帧概率大于历史最高时，更新历史最高字典结果为本帧结果
                            his_maxacc = {'max_name': names[0], 'max_sim': sims[0]}
                        else:  # 否则不更新历史最高结果
                            pass
                else:  # 同人，不清晰，不更新历史最高结果表
                    pass
                names[0] = his_maxacc['max_name']  # 将此人历史最高结果付给names作为返回
                # names[0] = his_maxacc['max_name'] + '_rg'+names[0]
            else:
                # 如果换人了，则将单人历史最准记录置零，再开始新的人的从0更新
                his_maxacc = {'max_name': '0-历史最准名默认值-0', 'max_sim': 0.0}

            # 单人模式实时流随机保存
            if is_qx1 >= qx_hold and is_qx0 >= qx_hold:
                sead = hash(str(time.time())[-6:])
                if sead % 2 == 1:  # hash采样
                    fpic_path = para_dct['savepic_path'] + 'stream/' + now_time + '_' + is_same_t + '-' + str(
                        is_same_p[0])[2:4] + '_' + str(int(is_qx1)) + '-' + str(int(is_qx0)) + '_' + str(
                        sims[0])[2:4]
                    cv2.imwrite(fpic_path + '_crop_@' + names[0] + '.jpg', crop_images_rg[0])
                    # cv2.imwrite(fpic_path + '_crop_@' + names[0] + '.jpg', crop_images_at[0])
                    cv2.imwrite(fpic_path + '_raw_@' + names[0] + '.jpg', f_pic)

            '''更新识别返回值员工id'''
            last_1p_emb = faceembs[0]  # 更新last save emb，以便判定本帧是否和上一帧同一个人
            ids_cut = [i.split('-')[0] for i in names]
        else:  # 多人，因为检测是识别两个过程，无法对应是否同人，所有目前没有比较好的结果修正，直接返回模型识别结果
            # 多人人脸识别，大于4的去掉不识别了【】
            names, faceis_konwns, faceembs, sims = facenet_pre_m.imgs_get_names(crop_images_rg, batchsize=len(dets))

            # 清晰度不符合的名字拍成‘不清晰’
            names_qx = []
            for i in range(len(dets)):
                is_qx1, is_qx0 = brenner(cv2.cvtColor(np.asarray(crop_images_rg[0], np.float32), cv2.COLOR_BGR2GRAY))
                if is_qx1 >= qx_hold and is_qx0 >= qx_hold:
                    names_qx.append(names[i])
                else:
                    names_qx.append('不清晰')
            names = names_qx

            # 按照距离左上角的远近返回名字顺序，越近越靠前显示。
            # dets_local = np.asarray([[i, (dets[i][0] * dets[i][0] + dets[i][1] * dets[i][1]) ** 0.5] for i in range(len(dets))], dtype=int)  # 计算矩形框离左上角的距离
            dets_local = np.asarray([[i, dets[i][0]] for i in range(len(dets))], dtype=int)  # 计算矩形框左上角x离video左上角的距离
            dets_local_rank = dets_local[dets_local[:, 1].argsort()]  # 按照距离排序输出名字顺序
            iter_n = range(len(dets))

            names_new = ['' for i in iter_n]
            faceis_konwns_new = ['' for i in iter_n]
            faceembs_new = ['' for i in iter_n]
            sims_new = ['' for i in iter_n]
            dets_new = ['' for i in iter_n]
            crop_images_at_new = ['' for i in iter_n]
            point5_at_new = ['' for i in iter_n]
            crop_images_rg_new = ['' for i in iter_n]
            point5_rg_new = ['' for i in iter_n]
            for i in range(len(names)):
                names_new[i] = names[dets_local_rank[i, 0]]
                faceis_konwns_new[i] = faceis_konwns[dets_local_rank[i, 0]]
                faceembs_new[i] = faceembs[dets_local_rank[i, 0]]
                sims_new[i] = sims[dets_local_rank[i, 0]]
                dets_new[i] = dets[dets_local_rank[i, 0]]
                crop_images_at_new[i] = crop_images_at[dets_local_rank[i, 0]]
                point5_at_new[i] = point5_at[dets_local_rank[i, 0]]
                crop_images_rg_new[i] = crop_images_rg[dets_local_rank[i, 0]]
                point5_rg_new[i] = point5_rg[dets_local_rank[i, 0]]
            names = names_new
            faceis_konwns = faceis_konwns_new
            faceembs = faceembs_new
            sims = sims_new
            dets = dets_new
            crop_images_at = crop_images_at_new
            point5_at = point5_at_new
            crop_images_rg = crop_images_rg_new
            point5_rg = point5_rg_new

            ids_cut = [i.split('-')[0] for i in names]
    else:  # 没有人
        ids_cut = ['无人' for i in dets]
        faceembs = [[] for i in dets]

    # 构造frame_rg_list 无人：[img视频原图]，有人：[img视频原图， {工号1: 33677，人脸图片: crop_img，向量: emb}, ...]
    frame_rg_list = [f_pic]
    if len(dets) != 0:
        for p_i in range(len(dets)):
            p1_rg_res = {'p1_id': ids_cut[p_i], 'p1_face': crop_images_at[p_i], 'p1_crop': crop_images_rg[p_i],
                         'p1_emb': faceembs[p_i]}
            frame_rg_list.append(p1_rg_res)
    else:  # 无人
        frame_rg_list.append({'p1_id': '无人', 'p1_face': [], 'p1_crop': [], 'p1_emb': []})

    # get_name_message('111')

    return frame_rg_list


def camera_open():
    camera1 = VideoStream(0)
    camera1.stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    camera1.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, c_w)
    camera1.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, c_h)
    camera1.start()
    return camera1


def background_thread():
    global camera, monitor_dct

    if not camera:
        camera = camera_open()

    i = 0  # 统计fps的时间
    start_flag = time.time()
    while 1:
        frame = camera.read()

        i += 1  # 统计fps的时间,计算每间隔了1s，会处理几张frame
        interval = int(time.time() - start_flag)
        if interval == 1:
            start_flag = time.time()
            i = 0

        # 每天23点00分的第一帧的时间点进行一次name_embs存储，因为可能有重名的问题，所以不能存为dict
        if (monitor_dct['add_n_all'] - monitor_dct['add_n_saved']) > 0:
            time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            if time_stamp[8:] in ['090000', '110000', '150000', '170000', '200000', '230000']:
                time.sleep(1.1)
                known_names_lst = [i[0] for i in facenet_pre_m.known_names]
                # print('aaaaaaaaaaaaaaaaaaaa', len(b))
                embds_dict = dict(zip(known_names_lst, facenet_pre_m.known_embs))
                with open(emb_pkl_path + time_stamp + "_embs_dict.pkl", 'wb') as f1:
                    pickle.dump(embds_dict, f1)
                print('@@@: add_n_saved embs pkl :', emb_pkl_path + time_stamp + "_embs_dict.pkl", 'wb')
                monitor_dct['add_n_saved'] = monitor_dct['add_n_all']

        if frame is not None:
            frame = cv2.flip(frame, 1)  # 前端输出镜面图片
            rg_1frame(frame)
        else:
            print('####################### camera not work !')
            camera = camera_open()  # 摄像头失效抓取为空，则重启摄像头
            continue


@app.route('/')
def index():
    return render_template('index_v2.html', async_mode=socketio.async_mode)


@socketio.on('connect', namespace='/test_conn')
def connect_message():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

    res_json = {'app_status': '1', 'app_data': {'message': '链接成功'}}
    emit('connect_response', res_json)


# @app.route("/cropimg/<user_id>")
# def get_crop_img(user_id):
#     imgPath = "丽江/{}.jpg".format(user_id)
#     with open(imgPath, 'rb') as f:
#         image = f.read()
#     return Response(image, mimetype='image/jpeg')


@socketio.on('get_name', namespace='/test_conn')  # 消息实时传送
def get_name_message(message):
    global api_status
    api_status = 'get_name_status'
    while api_status == 'get_name_status':
        print('---------------------------------------------------------1', api_status, ' monitor_dct:', monitor_dct)
        time.sleep(0.05)
        st = time.time()
        global frame_rg_list, all_officeinfo_dct
        res_json = {'app_data': {'message': '识别成功', 'persons': []}, 'app_status': '1'}
        p1_id = frame_rg_list[1]['p1_id']
        if p1_id not in ['不清晰', '无人', '历史最准名默认值']:
            for i in range(len(frame_rg_list)):
                if i == 0:  # list里第一项是原图，此处不需要，take photo的时候才需要
                    pass
                elif i in [1, 2, 3, 4]:
                    p1_id = frame_rg_list[i]['p1_id']
                    if p1_id != '0':
                        c_name = all_officeinfo_dct[p1_id][0]
                        e_name = all_officeinfo_dct[p1_id][1]
                        is_birth = is_birthday(all_officeinfo_dct[p1_id][2])
                    else:
                        c_name = '不认识'
                        e_name = '请点击下方按钮录入~'
                        is_birth, blessings = '0', ''
                    _, crop_raw_pic = cv2.imencode('.jpg', frame_rg_list[i]['p1_face'])
                    crop_img = crop_raw_pic.tobytes()
                    res_json['app_data']['persons'].append({'p1_id': p1_id, 'c_name': c_name, 'e_name': e_name,
                                                            'is_birth': is_birth, 'crop_img': crop_img})
                else:  # 只显示前人脸概率最大的前4个人
                    break
        else:
            res_json = {'app_data': {'message': '本帧无效', 'persons': []}, 'app_status': '0'}
        # print(sys.getsizeof(res_json), np.round(time.time() - st, 4))
        emit('get_name_response', res_json)


@socketio.on('get_video', namespace='/test_conn')  # 实时返回摄像头结果
def get_video_message(message):
    global api_status
    api_status = 'get_video_status'

    while api_status == 'get_video_status':
        print('---------------------------------------------------------2', api_status, ' monitor_dct:', monitor_dct)
        st = time.time()
        time.sleep(0.01)
        global camera
        if not camera:
            camera = camera_open()
        frame = camera.read()
        res_json = {'app_data': {'message': '获取实时帧成功'}, 'app_status': '1'}
        if frame is not None:
            frame = cv2.flip(frame, 1)  # 前端输出镜面图片
            _, raw_pic = cv2.imencode('.jpg', cv2.resize(frame, (
                int(c_w * para_dct['video_size_r']), int(c_h * para_dct['video_size_r']))))
            res_json['app_data']['video_pic'] = raw_pic.tobytes()
        else:
            res_json = {'app_data': {'message': '获取实时帧失败'}, 'app_status': '0'}

        # print(sys.getsizeof(res_json), np.round(time.time() - st, 4))
        emit('get_video_response', res_json)


@socketio.on('lock_video', namespace='/test_conn')  # 消息实时传送
def lock_video_message(message):
    st = time.time()
    global api_status
    api_status = 'lock_video_status'
    if api_status == 'lock_video_status':
        print('---------------------------------------------------------3', api_status)
        global frame_rg_list, photo_rg_list
        res_json = {'app_data': {'message': '图片有效'}, 'app_status': '1'}

        if len(frame_rg_list) == 2 and frame_rg_list[1]['p1_id'] not in ['无人', '不清晰']:
            photo_rg_list = frame_rg_list
            _, raw_pic = cv2.imencode('.jpg', cv2.resize(photo_rg_list[0], (
                int(c_w * para_dct['video_size_r']), int(c_h * para_dct['video_size_r']))))
            res_json['app_data']['video_pic'] = raw_pic.tobytes()
        else:
            res_json = {'app_data': {'message': '图片无效'}, 'app_status': '0'}
            photo_rg_list = [[], {'p1_id': '无人', 'p1_face': [], 'p1_crop': [], 'p1_emb': []}]
        # print(sys.getsizeof(res_json), np.round(time.time() - st, 4))

        emit('lock_video_response', res_json)


@socketio.on('add_new', namespace='/test_conn')  # 消息实时传送
def add_new_message(message):
    st = time.time()
    global api_status
    para = dict(message)
    api_status = 'add_new_status'
    if api_status == 'add_new_status':
        print('---------------------------------------------------------4', api_status)
        global frame_rg_list, photo_rg_list, monitor_dct
        p_id_input = para['P1']
        p_angle_input = para['P2']
        res_json = {'app_data': {'message': '录入成功'}, 'app_status': '1'}

        if p_id_input in all_officeinfo_dct.keys():
            if len(photo_rg_list) == 2 and photo_rg_list[1]['p1_id'] not in ['无人', '不清晰']:
                raw_pic = photo_rg_list[0]
                p1_crop = photo_rg_list[1]['p1_crop']
                p1_emb = photo_rg_list[1]['p1_emb']
                time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                monitor_dct['add_n_all'] += 1
                name_ps = time_stamp + p_angle_input + str(monitor_dct['add_n_all'])
                pic_name = p_id_input + '-' + all_officeinfo_dct[p_id_input][0] + '-' + name_ps

                facenet_pre_m.known_embs = np.row_stack((facenet_pre_m.known_embs, np.asarray(p1_emb)))
                facenet_pre_m.known_vms = np.row_stack((facenet_pre_m.known_vms, np.asarray([np.linalg.norm(p1_emb)])))
                facenet_pre_m.known_names = np.row_stack((facenet_pre_m.known_names, np.asarray([pic_name])))
                cv2.imwrite(para_dct['savepic_path'] + 'photos/' + pic_name + '_crop.jpg', p1_crop)
                cv2.imwrite(para_dct['savepic_path'] + 'photos/' + pic_name + '_raw.jpg', raw_pic)
                print('@@@: add 1 person', para_dct['savepic_path'] + 'photos/' + pic_name + '_crop.jpg')
            elif len(photo_rg_list) == 2 and photo_rg_list[0] == []:
                res_json['app_status'] = '3'
                res_json['app_data'] = {'message': '尚未拍照'}
            else:
                res_json['app_status'] = '2'
                res_json['app_data'] = {'message': '照片无效'}
        else:
            res_json['app_status'] = '0'
            res_json['app_data'] = {'message': '工号不存在'}
        # print(sys.getsizeof(res_json), np.round(time.time() - st, 4))

        photo_rg_list = [[], {'p1_id': '无人', 'p1_face': [], 'p1_crop': [], 'p1_emb': []}]  # 添加完信息后，把以保存的注空
        emit('add_new_response', res_json)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
    # nohup /usr/bin/python3 -u server_socket_old.py > /root/v2_face_rg/nohup.out 2>&1 &
    # tail -f /root/face_rg_server/nohup.out

    #
    # # 测试
    # with thread_lock:
    #     if thread is None:
    #         thread = socketio.start_background_task(background_thread)
    #
    # res_json = {'app_status': '1', 'app_data': {'message': '链接成功'}}
    # # res_json = json.dumps(res_json)
