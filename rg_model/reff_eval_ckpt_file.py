import tensorflow as tf
import argparse
from rg_net.net_insight_auroua_issue9 import get_resnet
import tensorlayer as tl
from rg_model.reff_verification import ver_test
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


if __name__ == '__main__':

    eval_datasets = ['agedb_30']  # evluation datasets
    eval_db_path = '../ver_data'  # evluate datasets base path
    image_size = [112, 112]  # the image size
    net_depth = 50  # resnet depth, default is 50
    num_output = 85164  # the image size
    batch_size = 32  # batch size to train network
    ckpt_file = '../auroua_intf/InsightFace_iter_best_'  # the ckpt file path
    ckpt_index_list = ['1950000.ckpt']  # ckpt file indexes

    ver_list = []
    ver_name_list = []
    for db in eval_datasets:
        print('begin db %s convert.' % db)
        data_set = 0  # (data_list, issame_list)，len(data_list)=2 ,data_list[0].shape=(12000, 112, 112, 3), len(issame_list) = 6000
        ver_list.append(data_set)  # [正(12000, 112, 112, 3), 反(12000, 112, 112, 3)], 6000
        ver_name_list.append(db)

    images = tf.placeholder(name='img_inputs', shape=[None, *image_size, 3], dtype=tf.float32)  # (?, 112, 112, 3)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)  # (?, )
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)  # 随机初始化权重先把空架子搭起来，后续再往里面restore train好的权重
    net = get_resnet(images, net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)  # L_Resnet_E_IR (?, 112,112,3)>(?, 512)
    embedding_tensor = net.outputs
    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss

    gpu_config = tf.ConfigProto()  
    gpu_config.gpu_options.allow_growth = True 
    sess = tf.Session(config=gpu_config)
    saver = tf.train.Saver()

    result_index = []
    for file_index in ckpt_index_list:
        feed_dict_test = {}
        path = ckpt_file +'/InsightFace_iter_best_'+'1950000'+'.ckpt'
        saver.restore(sess, path)
        print('ckpt file %s restored!' % file_index)
        feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
        feed_dict_test[dropout_rate] = 1.0
        results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
                           embedding_tensor=embedding_tensor, batch_size=batch_size, feed_dict=feed_dict_test,
                           input_placeholder=images)
        result_index.append(results)
    print(result_index)

