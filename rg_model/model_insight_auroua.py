import cv2
import glob
import os
import pickle
import numpy as np
from collections import Counter
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from rg_net import net_insight_lucky_resnetV2fix, net_insight_lucky_resnetV2
import math
from config_file import auroua_config_mgpu as cfg_au
from rg_net.net_insight_auroua_issue9 import get_resnet
import tensorlayer as tl
from data_pro.data_utils import load_image, data_iter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, classification_report


def get_embd(inputs, is_training_dropout, is_training_bn, config, reuse=False, scope='embd_extractor'):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        end_points = {}
        if config['backbone_type'].startswith('resnet_v2_m'):
            arg_sc = net_insight_lucky_resnetV2fix.resnet_arg_scope(weight_decay=config['weight_decay'],
                                                                    batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_m_50':
                    net, end_points = net_insight_lucky_resnetV2fix.resnet_v2_m_50(net, is_training=is_training_bn,
                                                                                   return_raw=True)
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
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_50(net, is_training=is_training_bn,
                                                                              return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_101':
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_101(net, is_training=is_training_bn,
                                                                               return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_152':
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_152(net, is_training=is_training_bn,
                                                                               return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_200':
                    net, end_points = net_insight_lucky_resnetV2.resnet_v2_200(net, is_training=is_training_bn,
                                                                               return_raw=True)
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


class InsightPreAuroua():
    def __init__(self):

        # 外部入参配置

        self.train_mode = 0
        self.au_cfg = cfg_au

        print('building net')

        self.images = tf.placeholder(name='img_inputs', shape=[None, *self.au_cfg.image_size, 3],
                                     dtype=tf.float32)  # (?, 112, 112, 3)
        self.labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)  # (?, )
        self.dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
        self.w_init_method = tf.contrib.layers.xavier_initializer(
            uniform=False)  # 随机初始化权重先把空架子搭起来，后续再往里面restore train好的权重
        self.net = get_resnet(self.images, self.au_cfg.net_depth, type='ir', w_init=self.w_init_method, trainable=False,
                              keep_rate=self.dropout_rate)  # L_Resnet_E_IR (?, 112,112,3)>(?, 512)
        self.embedding_tensor = self.net.outputs

        # 配置gpu，起session，restore参数
        print('restore model para')
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.feed_dict_test = {}
        print('ckpt file %s restored!' % self.au_cfg.premodel_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.au_cfg.premodel_path)
        self.feed_dict_test.update(tl.utils.dict_to_one(self.net.all_drop))
        self.feed_dict_test[self.dropout_rate] = 1.0

        # 实例化一次网络
        # load已知人脸
        self.files_fresh, self.known_names, self.known_embs, self.known_vms = None, None, None, None
        self.load_knows_pkl()

        exe_path = os.path.abspath(__file__)
        fontpath = str(exe_path.split('face_rg_server/')[0]) + 'face_rg_server/' + "data_pro/pre_img.jpg"
        image_pre1 = cv2.imread(fontpath)
        img_crop = np.asarray([cv2.resize(image_pre1, self.au_cfg.image_size)])
        face_names, is_knowns, face_embs, sim_pro_lst = self.imgs_get_names(img_crop)
        print('init done')

    def run_embds(self, crop_images, batch_size=1):
        all_embeddings = None
        self.feed_dict_test.setdefault(self.images, None)
        # for idx, data in enumerate(data_iter(crop_images, self.au_cfg.batch_size)):
        for idx, data in enumerate(data_iter(crop_images, batch_size)):
            print('batch n_th:', idx)
            data_tmp = np.asarray(data.copy(), dtype='float64')  # fix issues #4 <class 'tuple'>: (32, 112, 112, 3)
            data_tmp -= 127.5
            data_tmp *= 0.0078125
            self.feed_dict_test[self.images] = data_tmp
            _embeddings = self.sess.run(self.embedding_tensor, self.feed_dict_test)
            if all_embeddings is None:
                all_embeddings = _embeddings
            else:
                all_embeddings = np.row_stack((all_embeddings, _embeddings))
        return all_embeddings

    @staticmethod
    def is_newest(model_path, init_time):
        current_time = os.path.getctime(model_path)
        return init_time != None and current_time == init_time

    def load_knows_pkl(self):
        # load 最新已知人脸pkl
        print(global_pkl_path)
        self.files_fresh = sorted(glob.iglob(global_pkl_path), key=os.path.getctime, reverse=True)[0]
        fr = open(self.files_fresh, 'rb')
        piccode_path_dct = pickle.load(fr)  # key 043374-人力资源部-张晓宛
        self.known_names = np.asarray(list(piccode_path_dct.keys()))
        self.known_embs = np.asarray(list(piccode_path_dct.values()))
        # 计算已知人脸向量的摩长,[|B|= reshape( (N,), (N,1) ) ]，以便后边的计算实时流向量，计算最相似用户时用
        self.known_vms = np.reshape(np.linalg.norm(self.known_embs, axis=1), (len(self.known_embs), 1))

        peoples = [i.split('-')[0] for i in self.known_names]
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

        # print(cos_sim)
        is_known = 0
        sim_p = max(cos_sim)
        if sim_p >= 0.65:  # 越大越严格
            loc_similar_most = np.where(cos_sim == sim_p)
            print(loc_similar_most)
            is_known = 1
            print('识别到最相似的人是：', sim_p, self.known_names[loc_similar_most][0])
            return self.known_names[loc_similar_most][0], is_known, sim_p
        else:
            loc_similar_most = np.where(cos_sim == sim_p)
            print('未识别到但最相似的人是：', sim_p, self.known_names[loc_similar_most][0])
            return '未知的同学', is_known, sim_p

    def gen_knowns_db(self, pic_path, pkl_path):

        # 读marking人脸图片list
        imgs_pic, fns = load_image(pic_path, self.au_cfg.image_size)

        # 获取embs
        print('forward running...')
        embds_arr = self.run_embds(imgs_pic, 128)
        embds_dict = dict(zip(fns, list(embds_arr)))

        if len(embds_arr) != 0:
            # 存已知人脸embs dict
            with open(pkl_path, 'wb') as f:
                pickle.dump(embds_dict, f)
            print('saving knows pkl...', len(embds_arr), pkl_path)

    def imgs_get_names(self, crop_image):
        # print('rg_start', len(crop_image))
        embds_arr = self.run_embds(crop_image, 64)
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

    def verify_db(self, konwn_path, unkonwn_path, print_flag=0):

        fr = open(konwn_path, 'rb')
        k_names_embs = pickle.load(fr)
        k_embs = np.asarray(list(k_names_embs.values()))
        k_names = [i.split('@')[-1].split('-')[0] for i in list(k_names_embs.keys())]

        fr = open(unkonwn_path, 'rb')
        uk_names_embs = pickle.load(fr)  # key 043374-人力资源部-张晓宛
        uk_all_sim = []
        uk_all_label = []
        for uk_name, uk_emb in uk_names_embs.items():
            # 计算每一个未知人脸的和所有m已知人脸的相似度值。产出相似度矩阵 [uk_n, k_m]
            v = np.reshape(uk_emb, (len(uk_emb), 1))
            uk_sims = self.d_cos(v, vs=k_embs)
            uk_all_sim.append(uk_sims)

            uk_name = uk_name.split('@')[-1].split('-')[0]
            uk_labels = []
            for k_name in k_names:
                if uk_name == k_name:
                    uk_labels.append(1)
                else:
                    uk_labels.append(0)
            uk_all_label.append(uk_labels)

        uk_all_sim = np.asarray(uk_all_sim)
        uk_all_label = np.asarray(uk_all_label)

        uk_all_sim = np.ravel(uk_all_sim)
        uk_all_label = np.ravel(uk_all_label)
        print(uk_all_sim.shape, uk_all_label.shape)

        auc_v = np.round(roc_auc_score(uk_all_label, uk_all_sim, average='micro'), 4)
        print('AUC 为:', auc_v)
        print('\n')
        if print_flag == 1:
            for hold_v in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
                uk_all_sim_01 = [i >= hold_v for i in uk_all_sim]

                # report = classification_report(uk_all_label, uk_all_sim_01)
                # print(report)
                matrix = confusion_matrix(uk_all_label, uk_all_sim_01)
                print('hold_v >=', hold_v)
                print(matrix[0][0])
                print(matrix[0][1])
                print(matrix[1][0])
                print(matrix[1][1])
                print('\n\n')

        # Compute ROC curve and ROC area for each class
        fpr, tpr, threshold = roc_curve(uk_all_label, uk_all_sim)  ###计算真正率和假正率

        return fpr, tpr, threshold


global_pkl_path = '/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_insight_auroua/*'

# if __name__ == "__main__":
#     '''产生k&uk的embs'''
#     insight_c = InsightPreAuroua()
#     root_path = '/Users/finup/Desktop/rg/ver_data/'
#     pk_file = root_path + 'dc_marking_trans_avg_k/'
#     pu_file = root_path + 'dc_marking_trans_avg_uk/'
#     pkl_pkg = '/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_insight_auroua/'
#
#     # ckpt_pt = '/Users/finup/Desktop/rg/face_rg_files/premodels/pm_insight_auroua/1107_auroua_out_office_avg/mgpu_res/ckpt/InsightFace_iter_'
#     # p_dct = {'50w29': '500150-29', '50w39': '500200-39', '50w49': '500250-49', '50w59': '500300-59',
#     #          '50w69': '500350-69', '50w79': '500400-79', '50w89': '500450-89', '50w99': '500500-99',
#     #          '50w109': '500550-109', '50w119': '500600-119'}
#     ckpt_pt = '/Users/finup/Desktop/rg/face_rg_files/premodels/pm_insight_auroua/1030_auroua_out/mgpu_res/ckpt/InsightFace_iter_'
#     p_dct = {'50w': '500000'}
#     for k, v in p_dct.items():
#         saver = tf.train.Saver()
#         saver.restore(insight_c.sess, ckpt_pt + v + '.ckpt')
#         k_pkl = pkl_pkg + k + '_office_aurora_asian_finetune_avg_k.pkl'
#         u_pkl = pkl_pkg + k + '_office_aurora_asian_finetune_avg_uk.pkl'
#         insight_c.gen_knowns_db(pk_file, k_pkl)
#         insight_c.gen_knowns_db(pu_file, u_pkl)
#
#     '''评估'''
#     pkl_pkg = '/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_insight_auroua/'
#     roc_fig = '/Users/finup/Desktop/rg/face_rg_server/data_pro/aurora_asian_finetune_roc.jpg'
#     plt.figure()
#     lw = 2
#     plt.figure(figsize=(12, 20))
#     colors = ['black', 'red', 'orange', 'yellow', 'green', 'c', 'deepskyblue', 'blue', 'darkviolet', 'deeppink', 'pink']
#     p_dct = {'50w': '500000', '50w29': '500150-29', '50w39': '500200-39', '50w49': '500250-49', '50w59': '500300-59',
#              '50w69': '500350-69', '50w79': '500400-79', '50w89': '500450-89', '50w99': '500500-99',
#              '50w109': '500550-109', '50w119': '500600-119'}
#     c_i = -1
#     for k, v in p_dct.items():
#         k_pkl = pkl_pkg + k + '_office_aurora_asian_finetune_avg_k.pkl'
#         u_pkl = pkl_pkg + k + '_office_aurora_asian_finetune_avg_uk.pkl'
#         fpr, tpr, threshold = insight_c.verify_db(k_pkl, u_pkl, print_flag=0)
#
#         roc_auc = auc(fpr, tpr)  # 计算auc的值
#         c_i += 1
#         plt.plot(fpr, tpr, color=colors[c_i],
#                  lw=lw, label=k + ': AUC' + str(np.round(roc_auc, 4)))  # 假正率为横坐标，真正率为纵坐标做曲线
#     plt.plot([0, 1], [0, 1.01], color='navy', lw=lw, linestyle='--')
#     plt.plot([0.1, 0.1], [0, 1.01], color='gray', linestyle='-.')
#     plt.plot([0.01, 0.01], [0, 1.01], color='gray', linestyle='-.')
#     plt.plot([0.001, 0.001], [0, 1.01], color='gray', linestyle='-.')
#     plt.xlim([0.0, 0.5])
#     plt.ylim([0, 1.01])
#     y_ticks = np.arange(0, 1.01, 0.01)
#     plt.yticks(y_ticks)
#     plt.grid(linestyle='-.', axis='y', which='major')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve @IDcard1315P')
#     plt.legend(loc="upper right")
#     plt.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.07, hspace=2, wspace=2)
#     plt.margins(0.03, 0.03)
#     plt.savefig(roc_fig)
