from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Block(namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list contains one (depth, depth_bottleneck, stride) tuple for each unit in the block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):  # (?,56,56,64)(?,28,28,128)(?,14,14,256)(?,7,7,512)
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)  # padding='VALID'


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])  # zero padding
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None, store_non_strided_activations=False, outputs_collections=None):
    current_stride = 1
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')
    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    return net


def resnet_arg_scope(weight_decay=0.0001,  # l2正则lambda权重系数
                     batch_norm_decay=0.9,  #  train的平滑momentum 原始的代码是在image net上训练的，decay设置的是0.999，这个数值越大，网络训练越平缓越慢，但是在小数据集上训练的时候可以选用较小的数值，比如0.99或者0.95，https://blog.csdn.net/qq_25737169/article/details/79048516
                     batch_norm_epsilon=2e-5,  # 避免除0
                     batch_norm_scale=True,  # 尺度缩放
                     activation_fn=tf.nn.leaky_relu,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': batch_norm_updates_collections,
        'fused': None,  # Use fused batch norm if possible.
        'param_regularizers': {'gamma': slim.l2_regularizer(weight_decay)},
    }

    with slim.arg_scope(  # 卷积、bn、pooling
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),  # 初始化器是用来保持每一层的梯度大小都差不多相同，使用uniform或者normal分布来随机初始化
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
