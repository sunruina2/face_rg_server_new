import time
# for i in range(1000000000):
#     a = time.time()
#     stra = time.strftime("%Y%m%d%H%M%S", time.localtime())
#     print('/n')
#     print(stra, a)
#     print('aaa')
#
#     sead = hash(str(time.time())[-6:])
#     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', sead , sead % 2)
#     if sead % 2 == 1:  # hash采样
#         print('bbb')
#         print(str(time.time())[-6:])


# import cv2
#
# cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output1.avi', fourcc, 20, (1920, 1080))
#
# i = 0
# start_flag = time.time()
#
# while cap.isOpened():
#     rval, frame = cap.read()
#     i += 1
#     interval = int(time.time() - start_flag)
#     if interval == 1:  # 计算每间隔了1s，会处理几张frame
#         print('#########################################################', i)
#         start_flag = time.time()
#         i = 0
# #    cv2.imshow("capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# a = [1.2, 3.4, 6,5,7]
# b= ''
# for i in a:
#     b+= str(int(i)) + '-'
# c = b[0:-1]
# print(b)
# print(c)

# for xi in range(10):
#     print(xi)
#
# import glob
# import os
#
# files_fresh = sorted(glob.iglob('../facenet_files/embs_pkl/*'), key=os.path.getctime, reverse=True)[0]
# print(files_fresh)


# import collections
#
# # d1 = {}
# d1 = collections.OrderedDict()  # 将普通字典转换为有序字典
# d1['a'] = 'A'
# d1['b'] = 'B'
# d1['c'] = 'C'
# d1['d'] = 'D'
# for k, v in d1.items():
#     print(k, v)


# import numpy as np
#
# def brenner(img_i):
#     img_i = np.asarray(img_i, dtype='float64')
#     x, y = img_i.shape
#     img_i -= 127.5
#     img_i *= 0.0078125  # 标准化
#     center = img_i[0:x - 2, 0:y - 2]
#     center_xplus = img_i[2:, 0:y - 2]
#     center_yplus = img_i[0:x - 2:, 2:]
#     Dx = np.sum((center_xplus - center) ** 2)
#     Dy = np.sum((center_yplus - center) ** 2)
#     return Dx, Dy
#
#
# dx, dy = brenner(
#     [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35], [41, 42, 43, 44, 55]])
# print(dx, dy)


# # 生成器可以再大数据环境下减小内存
# # yeild可以把函数变成生成器
# # 用 yeild替换 return
#
# import time
#
#
# def countnum():
#     for i in range(10):
#         # return i
#         print('nananana')
#         yield i
#
#
# a = countnum()
# print(next(a))
# print(next(a))
# print(next(a))
# time.sleep(5)
# print(next(a))


# import time
#
#
# def sleeptime(hour, min, sec):
#     return hour * 3600 + min * 60 + sec
#
#
# second = sleeptime(0, 0, 1)
# while 1 == 1:
#     time.sleep(second)
#     print('do action')


# 单层目录+文件夹
# import os
# import shutil
# pic_path = '/Users/finup/Desktop/rg/员工照/所有员工face160'
# for root, dirs, files in os.walk(pic_path):
#     print(root)
#     for pic in files:
#         people_name = pic.split('.')[0]
#         if people_name != '':
#             srcfile = pic_path+'/'+people_name
#             dstfile = pic_path+'/'+people_name
#             os.mkdir(dstfile)
#             print(srcfile+'.png')
#             print(dstfile+'/'+people_name+'.png')
#             shutil.move(srcfile+'.png', dstfile+'/'+people_name+'.png')


# # 切除轴坐标边框取人脸
# import os
# import cv2
# pic_path = '/Users/finup/Desktop/rg/facenet_files/office_face160'
# for root, dirs, files in os.walk(pic_path):
#     for pic in files:
#         people_name = pic.split('.')[0]
#         if people_name != '':
#             print(pic_path+'/'+people_name+'/'+people_name+'.png')
#             img = cv2.imread(pic_path+'/'+people_name+'/'+people_name+'.png')
#             img_160 = img[59: 427, 145: 513, :]
#
#             cv2.imwrite(pic_path+'/'+people_name+'/'+people_name+'.png', img_160)


# a = 3*3 * (3*10+10*16+16*32 + 3*28+28*48+48*64 + 3*32+32*64+64*64+64*128) + (3*3*16*32)+ (3*3*64*128) + (3*3*128*256) + (32+128+256)*16
# print(a)


# # coding:utf8
# import tensorflow as tf
# import numpy as np
#
# sess = tf.InteractiveSession()
# x = tf.constant([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 4], [41, 42, 43, 44]])
# t = tf.convert_to_tensor(x)
# print("tensor", "*" * 16)
# print(t.eval())
# print("split x_axix", "*" * 10)
# # split第一个参数指定的沿某轴进行分割，
# # 第二个参数分成几个，tensor再这个方向上的维度值应该能被此参数整除
# print(tf.split(0, 2, t))
# print("*" * 23)
# print(tf.split(0, 2, t)[1].eval())
# print("split y_axix", "*" * 10)
# for i in range(4):
#     print(tf.split(1, 4, t)[i].eval())
# sess.close()

