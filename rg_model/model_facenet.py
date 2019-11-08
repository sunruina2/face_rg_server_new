import tensorflow as tf
from rg_net import net_facenet
import numpy as np
import pickle
import cv2
import os
import time
from collections import Counter
import glob
from data_pro.data_utils import load_image, data_iter


class FacenetPre():
    def __init__(self):

        self.model_dir = '/Users/finup/Desktop/rg/face_rg_files/premodels/pm_facenet'
        self.embs_dir = '/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_facenet'

        # gpu设置
        gpu_config = tf.ConfigProto()
        gpu_config.allow_soft_placement = True
        gpu_config.gpu_options.allow_growth = True
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        self.sess = tf.Session(config=gpu_config)
        net_facenet.load_model(self.sess, self.model_dir)
        # 返回给定名称的tensor
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print('建立facenet embedding模型')


        # load已知人脸
        self.files_fresh, self.known_names, self.known_embs, self.known_vms = None, None, None, None
        self.load_knows_pkl()

        # image_pre = cv2.imread('data_pro/pre_img.jpg')  # 首次run sess比较耗时，因此在初始化的时候预使用一张sample照片，使得线上实时流不受首次影响而延迟
        # crop_image = np.asarray([cv2.resize(image_pre, (160, 160))])
        # face_embs = self.sess.run(self.embeddings,
        #                           feed_dict={self.images_placeholder: crop_image, self.phase_train_placeholder: False})
        # print('init..')


    @ staticmethod
    def is_newest(model_path, init_time):
        current_time = os.path.getctime(model_path)
        return init_time != None and current_time == init_time

    def load_knows_pkl(self):
        # load 最新已知人脸pkl
        self.files_fresh = sorted(glob.iglob(self.embs_dir+'/*'), key=os.path.getctime, reverse=True)[0]
        print(self.files_fresh)
        with open(self.files_fresh, 'rb') as fr:
            piccode_path_dct = pickle.load(fr)
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

    def emb_toget_name(self, detect_face_embs_i, known_names_i, known_embs_i):  # 一张脸进来
        cos_sim = self.d_cos(detect_face_embs_i)
        is_known = 0
        sim_p = max(cos_sim)
        if sim_p >= 0.75:  # 越大越严格
            loc_similar_most = np.where(cos_sim == sim_p)
            is_known = 1
            return known_names_i[loc_similar_most][0], is_known, sim_p
        else:
            loc_similar_most = np.where(cos_sim == sim_p)
            # print('未识别到但最相似的人是：', sim_p, known_names_i[loc_similar_most][0])
            return '未知的同学', is_known, sim_p

    def emb_toget_name_old(self, detect_face_embs_i, known_names_i, known_embs_i):  # 一张脸进来
        L2_dis = np.linalg.norm(detect_face_embs_i - known_embs_i, axis=1)
        is_known = 0
        sim_p = min(L2_dis)
        if sim_p < 0.6:  # 越小越严格
            loc_similar_most = np.where(L2_dis == sim_p)
            is_known = 1
            return known_names_i[loc_similar_most][0], is_known, sim_p
        else:
            loc_similar_most = np.where(L2_dis == sim_p)
            print('未识别到但最相似的人是：', sim_p, known_names_i[loc_similar_most][0])
            return '未知的同学', is_known, sim_p

    def gen_knowns_db(self, pic_path, pkl_name):

        # 读marking人脸图片list
        imgs_pic, fns = load_image(pic_path, (160, 160))

        embds_arr = self.run_embds(imgs_pic, 64)
        embds_dict = dict(zip(fns, list(embds_arr)))

        if len(embds_dict) != 0:
            # 存已知人脸embs dict
            with open(pkl_name, 'wb') as f:
                pickle.dump(embds_dict, f)
            print('saving knows pkl...', len(embds_dict), pkl_name)

    @staticmethod
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)  # 图像归一化处理
        return y

    def imgs_get_names(self, crop_image):
        # print('rg_start', len(crop_image))
        # crop_image_nor = []
        # for aligned_pic in range(len(crop_image)):
        #     prewhitened = self.prewhiten(crop_image[aligned_pic])
        #     crop_image_nor.append(prewhitened)
        # crop_image_nor = np.stack(crop_image_nor)
        # face_embs = self.sess.run(self.embeddings,
        #                           feed_dict={self.images_placeholder: crop_image_nor,
        #                                      self.phase_train_placeholder: False})
        # print('rg_emb_ok')

        face_embs = self.run_embds(crop_image, 1)
        # face_embs = face_embs / np.linalg.norm(face_embs, axis=1, keepdims=True)  # 然后再求方向向量


        face_names = []
        is_knowns = []
        sim_pro_lst = []

        fresh_pkl = sorted(glob.iglob(self.embs_dir+'/*'), key=os.path.getctime, reverse=True)[0]
        if fresh_pkl != self.files_fresh:
            print(fresh_pkl)
            print(self.files_fresh)
            self.load_knows_pkl()
        for face_k in range(len(face_embs)):
            face_name, is_known, sim_pro = self.emb_toget_name(face_embs[face_k], self.known_names, self.known_embs)
            face_names.append(face_name)
            is_knowns.append(is_known)
            sim_pro_lst.append(sim_pro)
        # print('rg_choose_ok')

        return face_names, is_knowns, face_embs, sim_pro_lst

    def verify_db(self, konwn_path, unkonwn_path):

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

        from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

        auc = np.round(roc_auc_score(uk_all_label, uk_all_sim, average='micro'),4)
        print('AUC 为:', auc)
        print('\n')

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


    def run_embds(self, crop_images, batch_size=1):
        all_embeddings = None
        # for idx, data in enumerate(data_iter(crop_images, self.au_cfg.batch_size)):
        for idx, data in enumerate(data_iter(crop_images, batch_size)):
            print('batch n_th:', idx)
            data_tmp = np.asarray(data.copy(), dtype='float64')  # fix issues #4 <class 'tuple'>: (32, 112, 112, 3)
            data_tmp -= 127.5
            data_tmp *= 0.0078125

            face_embs = self.sess.run(self.embeddings, feed_dict={self.images_placeholder: data_tmp,
                                                 self.phase_train_placeholder: False})
            if all_embeddings is None:
                all_embeddings = face_embs
            else:
                all_embeddings = np.row_stack((all_embeddings, face_embs))
        return all_embeddings


if __name__ == "__main__":
    facenet_c = FacenetPre()

    time_stamp_pkl = time.strftime("%Y%m%d%H%M%S", time.localtime())
    facenet_c.gen_knowns_db('/Users/finup/Desktop/rg/face_rg_files/common_files/dc_marking_1known_trans',
                            '/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_facenet/' + time_stamp_pkl +'_facenet-known.pkl')
    facenet_c.gen_knowns_db('/Users/finup/Desktop/rg/face_rg_files/common_files/dc_marking_trans',
                            '/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_facenet/' + time_stamp_pkl +'_facenet-unknown.pkl')


    konwn_path='/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_facenet/'+time_stamp_pkl+'_facenet-known.pkl'
    unkonwn_path = '/Users/finup/Desktop/rg/face_rg_files/embs_pkl/ep_facenet/'+time_stamp_pkl+'_facenet-unknown.pkl'
    facenet_c.verify_db(konwn_path, unkonwn_path)


