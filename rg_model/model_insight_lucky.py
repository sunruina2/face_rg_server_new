import cv2
import glob
import os
import yaml
import pickle
import numpy as np
from collections import Counter
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from rg_net import net_insight_lucky_resnetV2fix, net_insight_lucky_resnetV2
import math


def get_embd(inputs, is_training_dropout, is_training_bn, config, reuse=False, scope='embd_extractor'):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        end_points = {}
        if config['backbone_type'].startswith('resnet_v2_m'):
            arg_sc = net_insight_lucky_resnetV2fix.resnet_arg_scope(weight_decay=config['weight_decay'],
                                                                    batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_m_50':
                    net, end_points = net_insight_lucky_resnetV2fix.resnet_v2_m_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_101':
                    net, end_points = net_insight_lucky_resnetV2fix.resnet_v2_m_101(net, is_training=is_training_bn,
                                                                                    return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_152':
                    net, end_points = net_insight_lucky_resnetV2fix.resnet_v2_m_152(net, is_training=is_training_bn,
                                                                                    return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_200':
                    net, end_points = net_insight_lucky_resnetV2fix.resnet_v2_m_200(net, is_training=is_training_bn,
                                                                                    return_raw=True)
                else:
                    raise ValueError('Invalid backbone type.')
        elif config['backbone_type'].startswith('resnet_v2'):
            arg_sc = net_insight_lucky_resnetV2.resnet_arg_scope(weight_decay=config['weight_decay'],
                                                                 batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_50':
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_101':
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_152':
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_200':
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_200(net, is_training=is_training_bn, return_raw=True)
        else:
            raise ValueError('Invalid backbone type.')

        if config['out_type'] == 'E':
            with slim.arg_scope(arg_sc):
                net = slim.batch_norm(net, activation_fn=None, is_training=is_training_bn)
                net = slim.dropout(net, keep_prob=config['keep_prob'], is_training=is_training_dropout)
                net = slim.flatten(net)  # (?, 25088)
                net = slim.fully_connected(net, config['embd_size'], normalizer_fn=None, activation_fn=None)  # (?, 512)
                net = slim.batch_norm(net, scale=False, activation_fn=None, is_training=is_training_bn)  # (?, 512)
                end_points['embds'] = net
        else:
            raise ValueError('Invalid out type.')

        return net, end_points


def pic1_process(face_pic, img_size):
    img = cv2.resize(face_pic, (img_size, img_size))
    img_f = np.fliplr(img)
    img = img / 127.5 - 1.0
    img_f = img_f / 127.5 - 1.0

    return img, img_f


def load_image(pics_path, image_size):
    print('pic reading %s' % pics_path)
    if os.path.isdir(pics_path):
        paths = list(os.listdir(pics_path))
        if '.DS_Store' in paths:  # 去掉不为文件夹格式的mac os系统文件
            paths.remove('.DS_Store')
    else:
        paths = [pics_path]
    images = []
    images_f = []
    fns = []
    for peo in paths:
        peo_i = 0
        peo_pics = list(os.listdir(pics_path + peo + '/'))
        if '.DS_Store' in peo_pics:  # 去掉不为jpg和png格式的mac os系统文件
            peo_pics.remove('.DS_Store')
        for pic in peo_pics:
            peo_i += 1
            p_path = pics_path + peo + '/' + pic
            # print(peo, peo_i, p_path)
            img_crop = cv2.imread(p_path)
            img, img_f = pic1_process(img_crop, image_size)
            images.append(img)
            images_f.append(img_f)

            time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            image_name = time_stamp + '_manualselected@' + peo.replace('-', '').replace('@160', '') + '-' + str(peo_i)
            fns.append(image_name)
    print('load pic done!', Counter(fns))
    return np.array(images), np.array(images_f), fns


class InsightPreLucky():
    def __init__(self):

        # 外部入参配置
        config_path = 'config_file/config_ms1m_200.yaml'
        model_path = '../facenet_files/premodel_insight_luck/config_ms1m_200_200k/best-m-200000'
        # model_path = '/Users/finup/Desktop/rg/facenet_files/premodel_insight_luck/config_ms1m_200_200k/best-m-200000'
        self.train_mode = 0
        self.yaml_cfg = yaml.load(open(config_path))

        print('building net')
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.yaml_cfg['image_size'], self.yaml_cfg['image_size'], 3],
                                     name='input_image')
        self.train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
        self.train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
        self.embeddings, _ = get_embd(self.images, self.train_phase_dropout, self.train_phase_bn, self.yaml_cfg)

        # 配置gpu，起session，restore参数
        print('restore model para')
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        tf.global_variables_initializer().run(session=self.sess)

        # 配置参数，预测时BN层参数使用预训练存好的均值方差
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # print(tf.trainable_variables(scope='embd_extractor/resnet_v2_50/block1/unit_2/block_v2/conv1/BatchNorm'))
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self.sess, model_path)

        # 实例化一次网络
        # load已知人脸
        self.files_fresh, self.known_names, self.known_embs, self.known_vms = None, None, None, None
        self.load_knows_pkl()
        print('init sample pic embd')
        img_crop = np.asarray([cv2.imread('data_pro/sample.jpg')])
        img_crop = cv2.resize(img_crop,(112, 112))
        face_names, is_knowns, face_embs, sim_pro_lst = self.imgs_get_names(img_crop)
        print(face_names, is_knowns)
        print(sim_pro_lst)
        print(face_embs)

        # img, img_f = pic1_process(img_crop, self.yaml_cfg['image_size'])
        # imgs_pic, imgs_f = np.asarray([img]), np.asarray([img_f]) # (n, 112,112,3)
        # embds_arr = self.run_embds(imgs_pic, self.yaml_cfg['batch_size'], self.yaml_cfg['image_size'])  # (n, 512)
        # embds_f_arr = self.run_embds(imgs_f, self.yaml_cfg['batch_size'], self.yaml_cfg['image_size'])  # (n, 512)
        # embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr / np.linalg.norm(
        #     embds_f_arr, axis=1, keepdims=True)  # 原图emb后的方向向量+左右翻转图emb后的方向向量，这样操作有什么作用呢？
        # embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)  # 然后再求方向向量
        # print(embds_arr)

    def run_embds(self, imgs_pic, batch_size, image_size):
        # self.sess, imgs_pic, yaml_cfg['batch_size'], yaml_cfg['image_size'], self.train_mode, self.embeddings, self.images,
        # self.train_phase_dropout, self.train_phase_bn
        if self.train_mode >= 1:
            is_trainning = True
        else:
            is_trainning = False
        batch_num = len(imgs_pic) // batch_size
        left = len(imgs_pic) % batch_size
        embds = []
        for i in range(batch_num):
            if batch_num > 1:
                print(i + 1, '/', batch_num)
            image_batch = imgs_pic[i * batch_size: (i + 1) * batch_size]
            cur_embd = self.sess.run(self.embeddings,
                                     feed_dict={self.images: image_batch, self.train_phase_dropout: is_trainning,
                                                self.train_phase_bn: is_trainning})
            embds += list(cur_embd)
        if left > 0:
            image_batch = np.zeros([batch_size, image_size, image_size, 3])  # (200,112,112,3)
            image_batch[:left, :, :, :] = imgs_pic[-left:]  # 把不足200的left余量放进，初始化好的 np.zeros 全为0的矩阵中去
            cur_embd = self.sess.run(self.embeddings,
                                     feed_dict={self.images: image_batch, self.train_phase_dropout: is_trainning,
                                                self.train_phase_bn: is_trainning})  # (200,512)
            embds += list(cur_embd)[:left]  # 取batch里的top前left个是真实的，后面的都是0的结果
        return np.array(embds)

    @staticmethod
    def is_newest(model_path, init_time):
        current_time = os.path.getctime(model_path)
        return init_time != None and current_time == init_time

    def load_knows_pkl(self):
        # load 最新已知人脸pkl
        self.files_fresh = sorted(glob.iglob(global_pkl_path), key=os.path.getctime, reverse=True)[0]
        fr = open(self.files_fresh, 'rb')
        piccode_path_dct = pickle.load(fr)  # key 043374-人力资源部-张晓宛
        self.known_names = np.asarray(list(piccode_path_dct.keys()))
        self.known_embs = np.asarray(list(piccode_path_dct.values()))
        # 计算已知人脸向量的摩长,[|B|= reshape( (N,), (N,1) ) ]，以便后边的计算实时流向量，计算最相似用户时用
        self.known_vms = np.reshape(np.linalg.norm(self.known_embs, axis=1), (len(self.known_embs), 1))

        peoples = [i.split('@')[1].split('-')[0] for i in self.known_names]
        count_p = Counter(peoples)
        print(count_p)
        print('已知人脸-总共有m个人:', len(list(set(peoples))))
        print('共计n个vectors:', len(self.known_names) - 1)
        print('平均每人照片张数:', int((len(self.known_names) - 1) / len(list(set(peoples)))))
        print('目前还有x人没有照片:', 61 - len(list(set(peoples))))

    # def d_cos_old(self, v):  # 输入需要是一张脸的v:(512,), knows_v:(N, 512)
    #     v = np.reshape(v, (1, len(v)))  # 变为1行
    #     num = np.dot(self.known_embs, v.T)  # (N, 1)
    #     denom = np.linalg.norm(v) * self.known_vms  # [|A|=float] * [|B|= reshape( (N,), (N,1) ) ] = (N, 1)
    #     cos = num / denom  # 余弦值 A * B / |A| * |B| 本身也是0-1之间的...
    #     # print('cos describe', max(cos), min(cos), np.mean(cos), np.var(cos))
    #     sim = 0.5 + 0.5 * cos  # 归一化到0-1之间, (N, 1)
    #     # print('sim describe', max(sim), min(sim), np.mean(sim), np.var(sim))
    #     sim = np.reshape(sim, (len(sim),))  # reshape((N,1), (N,)) 变成一维，方便后边算最大值最小值
    #
    #     return sim
    #
    # def d_cos(self, v):  # 输入需要是一张脸的v:(512,), knows_v:(N, 512)
    #     v = np.reshape(v, (1, len(v)))  # 变为1行
    #     dot = np.dot(self.known_embs, v.T)  # (N, 1)
    #     norm = np.linalg.norm(v.T) * self.known_vms
    #     similarity = dot / norm
    #     # dist = np.arccos(similarity) / math.pi
    #     similarity = np.reshape(similarity, (len(similarity),))
    #     return np.asarray(list(similarity))

    def d_cos(self, v, vs=None):  # 输入需要是一张脸的v:(512,) or (512,1), knows_v:(N, 512)
        if vs is None:
            vs = self.known_embs
            vs_norm = self.known_vms
        else:
            if len(vs.shape) == 1:
                vs = np.reshape(vs, (1, len(vs)))
                vs_norm = np.reshape(np.linalg.norm(vs, axis=1), (len(vs), 1))
            else:
                vs = vs
                vs_norm = np.reshape(np.linalg.norm(vs, axis=1), (len(vs), 1))

        v = np.reshape(v, (1, len(v)))  # 变为1行
        num = np.dot(vs, v.T)  # (N, 1)
        denom = np.linalg.norm(v) * vs_norm  # [|A|=float] * [|B|= reshape( (N,), (N,1) ) ] = (N, 1)
        cos = num / denom  # 余弦值 A * B / |A| * |B| 本身也是0-1之间的...

        sim = 0.5 + 0.5 * cos  # 归一化到0-1之间, (N, 1)

        # print('cos describe', max(cos), min(cos), np.mean(cos), np.var(cos))
        # print('sim describe', max(sim), min(sim), np.mean(sim), np.var(sim))
        sim = np.reshape(sim, (len(sim),))  # reshape((N,1), (N,)) 变成一维，方便后边算最大值最小值

        """
        人脸库中的照片pre_img.jpg，余弦距离参考值如下，有人脸图片cos均值在0.40842828，sim均值在 0.7042141，因此至少sim要大于0.70
        cos describe [0.99029934] [-0.07334533] 0.40842828 0.016055087
        sim describe [0.9951497] [0.46332735] 0.7042141 0.0040137717
        pre_1pic ['20190904205458_正面_024404-张佳丽'] [1] [0.9951497]

        无人脸的图片pre_bug.jpg，余弦距离参考值如下，无人脸有内容图片cos均值在0.11156807，sim均值在 0.55578405
        cos describe [0.47486433] [-0.09186573] 0.11156807 0.004270094
        sim describe [0.7374322] [0.45406714] 0.55578405 0.0010675235
        pre_1pic ['未知的同学'] [0.0] [0]

        近乎全白的图片pre_white.jpg，余弦距离参考值如下，白图cos均值在0.015752314，sim均值在 0.50787616
        cos describe [0.44681713] [-0.17200288] 0.015752314 0.00459828
        sim describe [0.7234086] [0.41399854] 0.50787616 0.0011495701
        pre_1pic ['未知的同学'] [0.0] [0]
        """

        return sim

    @staticmethod
    def distance(embeddings1, embeddings2, is_cos=0):
        if is_cos == 0:
            # Euclidian distance np.arange(0, 4, 0.01)
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff), 1)
        elif is_cos == 1:
            # Distance based on cosine similarity np.arange(0, 1, 0.0025)
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot / norm
            dist = np.arccos(similarity) / math.pi
        else:
            raise 'Undefined distance metric %d' % is_cos

        return dist

    def emb_toget_name(self, detect_face_embs_i):  # 一张脸进来
        cos_sim = self.d_cos(detect_face_embs_i)
        #############new #######?????????????距离计算需要改 ！！
        # cos_sim = []
        # for k_emb in self.known_embs:
        #     dis = self.distance(detect_face_embs_i, k_emb, is_cos=1)
        #     cos_sim.append(dis)
        # cos_sim = np.asarray(cos_sim)

        print(cos_sim)
        is_known = 0
        sim_p = max(cos_sim)
        if sim_p >= 0.75:  # 越大越严格
            loc_similar_most = np.where(cos_sim == sim_p)
            print(loc_similar_most)
            is_known = 1
            print('识别到最相似的人是：', sim_p, self.known_names[loc_similar_most][0])
            return self.known_names[loc_similar_most][0], is_known, sim_p
        else:
            loc_similar_most = np.where(cos_sim == sim_p)
            print('未识别到但最相似的人是：', sim_p, self.known_names[loc_similar_most][0])
            return '未知的同学', is_known, sim_p

    def gen_knows_db(self, pic_path, pkl_path):

        # 读marking人脸图片list
        imgs_pic, imgs_f, fns = load_image(pic_path, self.yaml_cfg['image_size'])

        # 获取embs
        print('forward running...')
        embds_arr = self.run_embds(imgs_pic, 128, self.yaml_cfg['image_size'])
        embds_f_arr = self.run_embds(imgs_f, 128, self.yaml_cfg['image_size'])
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr / np.linalg.norm(
            embds_f_arr, axis=1, keepdims=True)  # 原图emb后的方向向量+左右翻转图emb后的方向向量，这样操作有什么作用呢？
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)  # 然后再求方向向量

        if len(embds_arr) != 0:
            # 存已知人脸embs dict
            print('saving knows pkl...')
            time_stamp_pkl = time.strftime("%Y%m%d%H%M%S", time.localtime())
            pkl_name = pkl_path + time_stamp_pkl + '_knowns_db_color_insightluck.pkl'
            embds_dict = dict(zip(fns, list(embds_arr)))
            with open(pkl_name, 'wb') as f:
                pickle.dump(embds_dict, f)

    def imgs_get_names(self, crop_image):
        # print('rg_start', len(crop_image))
        crop_image_nor, crop_image_f_nor = [], []
        for aligned_pic in range(len(crop_image)):
            img, img_f = pic1_process(crop_image[aligned_pic], self.yaml_cfg['image_size'])
            crop_image_nor.append(img)
            crop_image_f_nor.append(img_f)
        crop_image_nor, crop_image_f_nor = np.asarray(crop_image_nor), np.asarray(crop_image_f_nor)  # (n, 112,112,3)

        embds_arr = self.run_embds(crop_image_nor, self.yaml_cfg['batch_size'], self.yaml_cfg['image_size'])  # (n, 512)
        embds_f_arr = self.run_embds(crop_image_f_nor, self.yaml_cfg['batch_size'],
                                     self.yaml_cfg['image_size'])  # (n, 512)
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr / np.linalg.norm(
            embds_f_arr, axis=1, keepdims=True)  # 原图emb后的方向向量+左右翻转图emb后的方向向量，这样操作有什么作用呢？
        face_embs = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True)  # 然后再求方向向量

        face_names = []
        is_knowns = []
        sim_pro_lst = []

        fresh_pkl = sorted(glob.iglob(global_pkl_path), key=os.path.getctime, reverse=True)[0]
        if fresh_pkl != self.files_fresh:
            self.load_knows_pkl()
        for face_k in range(len(face_embs)):
            face_name, is_known, sim_pro = self.emb_toget_name(face_embs[face_k])
            face_names.append(face_name)
            is_knowns.append(is_known)
            sim_pro_lst.append(sim_pro)
        # print('rg_choose_ok')

        return face_names, is_knowns, face_embs, sim_pro_lst


global_pkl_path = '../facenet_files/embs_pkl/insight_luck/*'

if __name__ == "__main__":
    insight_c = InsightPreLucky()
    # insight_c.gen_knows_db('/Users/finup/Desktop/rg/facenet_files/dc_marking_test/', '../source/ebms_pkl_lucky/')
