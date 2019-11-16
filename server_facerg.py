#! coding=utf-8
from flask import Flask, render_template, Response, redirect, request
import cv2
from imutils.video import VideoStream
import numpy as np
from align import mtcnn_def_trans as fc_server
import time
import pickle
from data_pro.data_utils import brenner
from anti.anti_pre import AntiSpoofing
anti = AntiSpoofing()

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

'''加载模型'''
facenet_pre_m = InsightPreAuroua()
lastsave_embs0 = np.zeros(512)
imgsize = 112
trans_01 = 1
pic_path = '/Users/finup/Desktop/rg/face_rg_files/common_files/dc_marking_trans_newnanme/'
pkl_path = '../face_rg_files/embs_pkl/ep_insight_auroua/50w_dc_all.pkl'
# facenet_pre_m.gen_knowns_db(pic_path, pkl_path)

'''设置全局变量'''
fr = open('../face_rg_files/common_files/officeid_name_dct.pkl', 'rb')
officeid_name_dct = pickle.load(fr)
names = []
faceembs = []
f_areas_r_lst = []
names1p_last_n = []
realtime = True
capture_saved = False
new_photo = None
save_flag = 0
add_faces_n = 0
app = Flask(__name__)
camera = None
c_w, c_h = 1280, 720
# c_w, c_h = 1920, 1080
mtcnn_minsize = int(0.2 * min(c_w, c_h))
last_exetime = time.time()
last_maxacc = {'max_name': '', 'max_sim': 0.0}


def rg_1frame(f_pic):
    global names
    global faceembs
    global f_areas_r_lst
    global names1p_last_n
    global lastsave_embs0, mtcnn_minsize, last_maxacc

    # 统计每步执行时间
    # last_exetime_rg = time.time()
    # now_exetime_rg = time.time()
    # print('TIME rg: ********************************************** Start', np.round((now_exetime_rg - last_exetime_rg), 4))
    # last_exetime_rg = now_exetime_rg

    det_para = [256, 0.5, 112, 0.0]  # 活体尺寸和延申%，识别尺寸和延伸%
    dets, crop_images_at, point5_at, crop_images_rg, point5_rg, align_flag = fc_server.load_and_align_data(f_pic,
                                                                                                           det_para,
                                                                                                           minsize=90)  # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片

    # now_exetime_rg = time.time()
    # print('TIME rg: aligin', np.round((now_exetime_rg - last_exetime_rg), 4))
    # last_exetime_rg = now_exetime_rg

    if align_flag != 0:
        is_qingxi1, is_qingxi0 = brenner(
            cv2.cvtColor(np.asarray(crop_images_rg[0], np.float32), cv2.COLOR_BGR2GRAY))  # is_qingxi1是枞向运动模糊方差，0是横向
        # now_exetime_rg = time.time()
        # print('TIME rg: qingxidu old filter', np.round((now_exetime_rg - last_exetime_rg), 4), is_qingxi1)
        # last_exetime_rg = now_exetime_rg

        # is_qingxi_p = cv2.Laplacian(cv2.cvtColor(crop_images[0], cv2.COLOR_BGR2GRAY), cv2.CV_16S, ksize=3).var()
        # now_exetime_rg = time.time()
        # # print('TIME rg: qingxidu new filter', np.round((now_exetime_rg - last_exetime_rg), 4), is_qingxi_p)
        # last_exetime_rg = now_exetime_rg
        tstr_pic = time.strftime("%Y%m%d%H%M%S", time.localtime())
        now_hour = int(tstr_pic[8:10])
        if now_hour >= 17:
            qx_hold = 50
        else:
            qx_hold = 100
        if is_qingxi1 >= qx_hold and is_qingxi0 >= qx_hold:  # 有人且清晰，则画人脸，进行识别名字
            names, faceis_konwns, faceembs, sims = facenet_pre_m.imgs_get_names(crop_images_rg,
                                                                                    batchsize=len(dets))  # 获取人脸名字
            # print(names)
            names = [i.split('-')[1] for i in names]
            # now_exetime_rg = time.time()
            # print('TIME rg: facenet rg', np.round((now_exetime_rg - last_exetime_rg), 4))
            # last_exetime_rg = now_exetime_rg

            # 抽样存储识别图片,图片命名：时间戳 + 是否和上一张同人 + 库中最相似的相似度 + 大小 + 识别名字结果
            if len(names) == 1:
                is_same_p = facenet_pre_m.d_cos(faceembs[0], lastsave_embs0)
                if is_same_p[0] > 0.80:
                    is_same_t = '1'
                else:
                    is_same_t = '0'
                # 单人模式 依据前后流进行最高概率修正

                if is_same_t == '1':
                    if sims[0] > last_maxacc['max_sim']:
                        # 如果遇到这个人更准确的脸部识别结果则用该最准结果作为输出结果。并把此刻最终结果存储在 历史最准字典中
                        last_maxacc = {'max_name': names[0], 'max_sim': sims[0]}
                        names[0] = names[0] + '_rg'+names[0]
                    else:
                        # 否则用这个人过去实时流中识别最准的识别结果。
                        names[0] = last_maxacc['max_name']
                        names[0] = last_maxacc['max_name'] + '_rg'+names[0]
                else:
                    # 如果换人了，则将单人历史最准记录置零，再开始新的人的从0更新
                    last_maxacc = {'max_name': '', 'max_sim': 0.0}

                # 单人照实时流随机保存
                sead = hash(str(time.time())[-6:])
                if sead % 2 == 1:  # hash采样
                    fpic_path = '../facenet_files/stream_pictures/' + tstr_pic + '_' + is_same_t + '-' + str(
                        is_same_p[0])[2:4] + '_' + str(int(is_qingxi1)) + '-' + str(int(is_qingxi0)) + '_' + str(
                        sims[0])[2:4]
                    cv2.imwrite(fpic_path + '_crop_' + names[0] + '.jpg', crop_images_rg[0])
                    # cv2.imwrite(fpic_path + '_raw_' + names[0] + '.jpg', f_pic)
                    lastsave_embs0 = faceembs[0]  # 更新last save emb，以便判定本帧是否和上一帧同一个人

                    # now_exetime_rg = time.time()
                    # print('TIME rg: save stream pic', np.round((now_exetime_rg - last_exetime_rg), 4))
                    # last_exetime_rg = now_exetime_rg

                    # 画鼻子眼睛保存，
                    # print('标记5点位置', point5[0])
                    # crop_img_mark = fc_server.mark_face_points(point5, crop_images[0])
                    # mark5 = ''
                    # for i in point5:
                    #     mark5 += str(int(i[0])) + '-'
                    # mark5 = mark5[0:-1]
                    # cv2.imwrite(fpic_path + '_cropmark' + '_' + mark5 + '_' + names[0] + '.jpg', crop_img_mark)

            '''活体检测'''
            anti_flag_list = []
            for i in range(len(dets)):
                anti_flag_list.append(anti(crop_images_at[i]))
            '''在画框上显示是否活体'''
            for i in range(len(dets)):
                names[i] = names[i]+'_'+str(anti_flag_list[i])

            # 绘制矩形框并标注文字
            f_pic, f_areas_r_lst = fc_server.mark_pic(dets, names, f_pic)

            # now_exetime_rg = time.time()
            # print('TIME rg: draw pic', np.round((now_exetime_rg - last_exetime_rg), 4))
            # last_exetime_rg = now_exetime_rg

            # #  对人脸名字进行稳定性修正
            # if len(names) == 5:
            #     if len(names1p_last_n) <= 1:  # 不到最近5帧继续追加
            #         names1p_last_n.append(names[0])
            #     else:  # 到5帧则去掉最早的，留下最近的
            #         del names1p_last_n[0]
            #         names1p_last_n.append(names[0])
            #
            #     top_names = Counter(names1p_last_n).most_common(2)  # 依据最近5帧统计top名字
            #     if len(top_names) >= 2:  # 如果名字种类两个名字以上
            #         name_top1 = top_names[0][0]
            #         name_top2 = top_names[1][0]
            #         if name_top1 == '0-未知的同学-0':  # 如果名字种类两个名字以上，且第一名是未知，则返回第二个，即已知
            #             names[0] = name_top2
            #         else:  # 如果名字种类两个名字以上，且Top1是已知，则返回Top1
            #             names[0] = name_top1

            return f_pic
        else:  # 有人但不清晰，则只画人脸
            names = ['抱歉清晰度不够^^' for i in dets]
            f_pic, f_areas_r_lst = fc_server.mark_pic(dets, names, f_pic)
            return f_pic
    else:  # 没有人
        return f_pic


def gen_frames():
    global camera, capture_saved, last_exetime
    global realtime, new_photo, save_flag, add_faces_n, c_w, c_h
    if not camera:
        camera = VideoStream(0)

        camera.stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
        camera.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, c_w)
        camera.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, c_h)
        camera.start()

    i = 0
    # 统计fps的时间
    start_flag = time.time()
    while 1:

        frame = camera.read()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg

        # 统计每步执行时间
        # now_exetime = time.time()
        # print('TIME:load>>gen', np.round((now_exetime - last_exetime), 4))
        # last_exetime = now_exetime

        i += 1
        interval = int(time.time() - start_flag)
        if interval == 1:  # 计算每间隔了1s，会处理几张frame
            print('#########################################################fps:', i, '  add_n:', add_faces_n)
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
            new_frame = rg_1frame(frame)

            # 统计每步执行时间
            # now_exetime = time.time()
            # print('TIME:gen>>mark', np.round((now_exetime - last_exetime), 4))
            # last_exetime = now_exetime

            if not realtime:
                if not capture_saved:
                    new_photo = frame
                    capture_saved = True
                _, jpeg = cv2.imencode('.jpg', new_photo)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                break
            else:
                _, jpeg = cv2.imencode('.jpg', new_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            # 统计每步执行时间
            # now_exetime = time.time()
            # print('TIME:mark>>dispaly', np.round((now_exetime - last_exetime), 4))
            # last_exetime = now_exetime
        else:
            # print('img_None')
            camera = VideoStream(0)  # 摄像头失效抓取为空，则重启摄像头
            continue


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html', txt="hello world")


@app.route('/video_feed')  # 这个地址返回视频流实时结果
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/txt')  # 每秒显示name变量的内容在右侧
def txt():
    global names, f_areas_r_lst, realtime
    # names = [i.split('@')[-1].split('-')[0] for i in names]
    return {'names': names, 'areas': f_areas_r_lst, 'realtime': str(realtime)}


@app.route('/add', methods=['POST'])  # 点击'录入'，更新人脸库dict k v，之后改为实时，并返回首页
def add():
    # update embedding
    global faceembs, realtime, new_photo, officeid_name_dct, add_faces_n, capture_saved
    # user_new = request.form["new_user"].replace(' ', '')
    if request.form['submit'] == 'yes':
        userid_new = request.form["new_user"].replace(' ', '').strip()
        if len(userid_new) > 0:
            if int(userid_new) in officeid_name_dct:
                user_new = officeid_name_dct[int(userid_new)]

                add_faces_n += 1
                time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                facenet_pre_m.known_embs = np.insert(facenet_pre_m.known_embs, 0,
                                                     values=np.asarray(faceembs[0]), axis=0)
                facenet_pre_m.known_vms = np.insert(facenet_pre_m.known_vms, 0,
                                                    values=np.linalg.norm(faceembs[0]), axis=0)

                addpic_name = request.form['cars'] + '-' + user_new + '-' + time_stamp
                facenet_pre_m.known_names = np.insert(facenet_pre_m.known_names, 0,
                                                      values=np.asarray(
                                                          addpic_name),
                                                      axis=0)
                cv2.imwrite(
                    "../facenet_files/photos/" + addpic_name + '.jpg',
                    new_photo)
            else:
                print('工号未知的新员工:', int(userid_new))

    realtime = True
    capture_saved = False
    return redirect('/')


@app.route('/capture', methods=['POST'])
def capture():
    # 单人入境时，点击拍下当前照片后，视频流锁住当前照片，
    # 并将状态改为非实时，此时gen_frames里面会保存图片并结束response的while循环
    global realtime
    realtime = False
    return redirect('/')


@app.route('/is_leave')
def is_leave():
    global f_areas_r_lst, realtime, capture_saved
    if len(f_areas_r_lst) == 0 or f_areas_r_lst[0] < 0.01:
        realtime = True
        capture_saved = False
    return str(realtime)


if __name__ == '__main__':
    app.run(host='localhost', port=8000)
