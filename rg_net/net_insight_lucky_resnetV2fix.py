from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from rg_net import net_insight_lucky_utils

resnet_arg_scope = net_insight_lucky_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.leaky_relu, scope='preact')
        if depth == depth_in:
            shortcut = net_insight_lucky_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = net_insight_lucky_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def block(inputs, depth, stride, rate=1, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'block_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.leaky_relu, scope='preact')  # (?,112,112,64)(?,56,56,64)(?,56,56,64)(?,56,56,64)(?,56,56,64)(?,28,28,128)(?,28,28,128)(?,28,28,128)(?,28,28,128)(?,14,14,256)(?,14,14,256)(?,14,14,256)(?,14,14,256)
        if depth == depth_in:
            shortcut = net_insight_lucky_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = net_insight_lucky_utils.conv2d_same(preact, depth, 3, stride, rate=rate, scope='conv1')
        residual = slim.conv2d(residual, depth, [3, 3], stride=1, normalizer_fn=None, activation_fn=None, scope='conv2')
        # residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2_m(inputs,
                blocks,
                num_classes=None,
                is_training=True,
                return_raw=True,
                global_pool=True,
                output_stride=None,
                include_root_block=True,
                spatial_squeeze=True,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, net_insight_lucky_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                        net = net_insight_lucky_utils.conv2d_same(net, 64, 3, stride=1, scope='conv1')
                    # net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = net_insight_lucky_utils.stack_blocks_dense(net, blocks, output_stride)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if return_raw:
                    return net, end_points
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                end_points[sc.name + '/postnorm'] = net

                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net

                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


resnet_v2_m.default_image_size = 224


def resnet_v2_bottleneck(scope, base_depth, num_units, stride):
    return net_insight_lucky_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }] + (num_units - 1) * [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }])


resnet_v2_m.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride):
    return net_insight_lucky_utils.Block(scope, block, [{
        'depth': base_depth * 4,
        'stride': stride  # block里第一次stride=2，先缩小，后加深
    }] + (num_units - 1) * [{
        'depth': base_depth * 4,
        'stride': 1
    }])


resnet_v2_m.default_image_size = 224


def resnet_v2_m_50(inputs,
                   num_classes=None,
                   is_training=True,
                   return_raw=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   reuse=None,
                   scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=16, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=32, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=64, num_units=14, stride=2),
        resnet_v2_block('block4', base_depth=128, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw,
                       global_pool=global_pool, output_stride=output_stride, include_root_block=True,
                       spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)


resnet_v2_m_50.default_image_size = resnet_v2_m.default_image_size


def resnet_v2_m_101(inputs,
                    num_classes=None,
                    is_training=True,
                    return_raw=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_bottleneck('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_bottleneck('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_bottleneck('block3', base_depth=256, num_units=23, stride=2),
        resnet_v2_bottleneck('block4', base_depth=512, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw,
                       global_pool=global_pool, output_stride=output_stride, include_root_block=True,
                       spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)


resnet_v2_m_101.default_image_size = resnet_v2_m.default_image_size


def resnet_v2_m_152(inputs,
                    num_classes=None,
                    is_training=True,
                    return_raw=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_bottleneck('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_bottleneck('block2', base_depth=128, num_units=8, stride=2),
        resnet_v2_bottleneck('block3', base_depth=256, num_units=36, stride=2),
        resnet_v2_bottleneck('block4', base_depth=512, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw,
                       global_pool=global_pool, output_stride=output_stride, include_root_block=True,
                       spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)


resnet_v2_m_152.default_image_size = resnet_v2_m.default_image_size


def resnet_v2_m_200(inputs,
                    num_classes=None,
                    is_training=True,
                    return_raw=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_bottleneck('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_bottleneck('block2', base_depth=128, num_units=24, stride=2),
        resnet_v2_bottleneck('block3', base_depth=256, num_units=36, stride=2),
        resnet_v2_bottleneck('block4', base_depth=512, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw,
                       global_pool=global_pool, output_stride=output_stride, include_root_block=True,
                       spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)


resnet_v2_m_200.default_image_size = resnet_v2_m.default_image_size

'''resnet50详细矩阵变换参考Tiny'''


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(
        2, 2)):  # x, 3, [64, 64, 256], stage=2, block='1a', strides=(2, 2), is_training=is_training, reuse=reuse
    filters1, filters2, filters3 = filters  # ResNet利用了1×1卷积，并且是在3×3卷积层的前后都使用了，不仅进行了降维，还进行了升维，使得卷积层的输入和输出的通道数都减小，参数数量进一步减少

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, strides=strides, name=conv_name_1,
                         reuse=reuse)  # 1a(?,56,56,64) >> 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)  # 1a(?,56,56,64)
    x = tf.nn.relu(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
    x = tf.layers.conv2d(x, filters2, kernel_size, padding='SAME', use_bias=False, name=conv_name_2,
                         reuse=reuse)  # 1a(?,56,56,64) >> 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)  # 1a(?,56,56,64)
    x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
    x = tf.layers.conv2d(x, filters3, (1, 1), name=conv_name_3, use_bias=False,
                         reuse=reuse)  # 1a(?,56,56,256) >> 2a(?,28,28,512)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3,
                                      reuse=reuse)  # 1a(?,56,56,256) >> 2a(?,28,28,512)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
    bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'
    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), use_bias=False, strides=strides, name=conv_name_4,
                                reuse=reuse)  # 1a(?,56,56,256) >> 2a(?,28,28,512)
    shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4,
                                             reuse=reuse)  # 1a(?,56,56,256) >> 2a(?,28,28,512)

    x = tf.add(shortcut, x)  # 对应元素相加，f(x) + x，# 1a(?,56,56,256)  2a(?,28,28,512)
    x = tf.nn.relu(x)
    return x


def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training,
                     reuse):  # x1, 3, [64, 64, 256], stage=2, block='1b', is_training=is_training, reuse=reuse
    filters1, filters2, filters3 = filters

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'

    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, name=conv_name_1,
                         reuse=reuse)  # 1b(?, 56, 56, 64) >> 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
    x = tf.nn.relu(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
    x = tf.layers.conv2d(x, filters2, kernel_size, padding='SAME', use_bias=False, name=conv_name_2,
                         reuse=reuse)  # 1b(?, 56, 56, 64) >> 2a(?,28,28,128)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
    x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
    x = tf.layers.conv2d(x, filters3, (1, 1), name=conv_name_3, use_bias=False,
                         reuse=reuse)  # 1b(?, 56, 56, 256) >> 2a(?,28,28,512)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

    x = tf.add(input_tensor, x)
    x = tf.nn.relu(x)
    return x


def resnet50(input_tensor, is_training=True, pooling_and_fc=True, reuse=False):
    x = tf.layers.conv2d(input_tensor, 64, (7, 7), strides=(1, 1), padding='SAME', use_bias=False,
                         name='conv1_1/3x3_s1', reuse=reuse)  # (?,112,112,64) 第一层卷积
    x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1',
                                      reuse=reuse)  # (?,112,112,64) bn 输入batch标准化
    x = tf.nn.relu(x)  # (?,112,112,64) 激活函数relu
    # x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), name='mpool1')

    x1 = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='1a', strides=(2, 2), is_training=is_training,
                       reuse=reuse)  # 先stride2一次减小，L:112>>56，D:3>>64升维,k=1；再卷一次L:56>>56，D:64>>64,k=3捕获像素八邻域信息；再卷一次加深L:56>>56，D:64>>256二次升维,k=1；123串行产生A，4input图上sdride2减小为fmap大小，L:112>>56，D:3>>256,k=1产生B；5A+B对应元素相加[升维，拓宽，升维，+x]
    x1 = identity_block2d(x1, 3, [64, 64, 256], stage=2, block='1b', is_training=is_training,
                          reuse=reuse)  # 三次卷积stride都=1，k分别为[1,3,1]，D分别为 [64,64,256]，[降维，拓宽，升维]
    x1 = identity_block2d(x1, 3, [64, 64, 256], stage=2, block='1c', is_training=is_training,
                          reuse=reuse)  # 三次卷积stride都=1，k分别为[1,3,1]，D分别为 [64,64,256]，[降维，拓宽，升维]

    x2 = conv_block_2d(x1, 3, [128, 128, 512], stage=3, block='2a', is_training=is_training, reuse=reuse)
    x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='2b', is_training=is_training, reuse=reuse)
    x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='2c', is_training=is_training, reuse=reuse)
    x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='2d', is_training=is_training,
                          reuse=reuse)  # (?,28,28,512)

    x3 = conv_block_2d(x2, 3, [256, 256, 1024], stage=4, block='3a', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3b', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3c', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3d', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3e', is_training=is_training, reuse=reuse)
    x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='3f', is_training=is_training,
                          reuse=reuse)  # (?,14,14,1024)

    x4 = conv_block_2d(x3, 3, [512, 512, 2048], stage=5, block='4a', is_training=is_training, reuse=reuse)
    x4 = identity_block2d(x4, 3, [512, 512, 2048], stage=5, block='4b', is_training=is_training, reuse=reuse)
    x4 = identity_block2d(x4, 3, [512, 512, 2048], stage=5, block='4c', is_training=is_training,
                          reuse=reuse)  # (?,7,7,2048)

    if pooling_and_fc:
        # pooling_output = tf.layers.max_pooling2d(x4, (7,7), strides=(1,1), name='mpool2')
        pooling_output = tf.contrib.layers.flatten(x4)  # (?, 100325=7*7*2058)
        fc_output = tf.layers.dense(pooling_output, 512, name='fc1', reuse=reuse)  # (?, 512)
        fc_output = tf.layers.batch_normalization(fc_output, training=is_training, name='fbn')

    return fc_output
