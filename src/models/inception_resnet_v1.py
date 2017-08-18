# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Inception-Renset-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Renset-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
  
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], 3)
    return net
  
def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None, 
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse) as res_scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                
                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net
        
                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net
                
                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net
                
                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net
                
                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net
                
                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net
                
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
          
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
          
                    end_points['PreLogitsFlatten'] = net
                
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)

    res_variables = tf.contrib.framework.get_variables(res_scope)
    return net, end_points, res_variables



'''
Variables: name (type shape) [size]
---------
InceptionResnetV1/Conv2d_1a_3x3/weights:0 (float32_ref 3x3x3x32) [864, bytes: 3456]
InceptionResnetV1/Conv2d_1a_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Conv2d_1a_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Conv2d_1a_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Conv2d_2a_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Conv2d_2a_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Conv2d_2a_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Conv2d_2a_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Conv2d_2b_3x3/weights:0 (float32_ref 3x3x32x64) [18432, bytes: 73728]
InceptionResnetV1/Conv2d_2b_3x3/BatchNorm/beta:0 (float32_ref 64) [64, bytes: 256]
InceptionResnetV1/Conv2d_2b_3x3/BatchNorm/moving_mean:0 (float32_ref 64) [64, bytes: 256]
InceptionResnetV1/Conv2d_2b_3x3/BatchNorm/moving_variance:0 (float32_ref 64) [64, bytes: 256]
InceptionResnetV1/Conv2d_3b_1x1/weights:0 (float32_ref 1x1x64x80) [5120, bytes: 20480]
InceptionResnetV1/Conv2d_3b_1x1/BatchNorm/beta:0 (float32_ref 80) [80, bytes: 320]
InceptionResnetV1/Conv2d_3b_1x1/BatchNorm/moving_mean:0 (float32_ref 80) [80, bytes: 320]
InceptionResnetV1/Conv2d_3b_1x1/BatchNorm/moving_variance:0 (float32_ref 80) [80, bytes: 320]
InceptionResnetV1/Conv2d_4a_3x3/weights:0 (float32_ref 3x3x80x192) [138240, bytes: 552960]
InceptionResnetV1/Conv2d_4a_3x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Conv2d_4a_3x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Conv2d_4a_3x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Conv2d_4b_3x3/weights:0 (float32_ref 3x3x192x256) [442368, bytes: 1769472]
InceptionResnetV1/Conv2d_4b_3x3/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Conv2d_4b_3x3/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Conv2d_4b_3x3/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Repeat/block35_1/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_1/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0c_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0c_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_1/Conv2d_1x1/weights:0 (float32_ref 1x1x96x256) [24576, bytes: 98304]
InceptionResnetV1/Repeat/block35_1/Conv2d_1x1/biases:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Repeat/block35_2/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_2/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0c_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0c_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_2/Conv2d_1x1/weights:0 (float32_ref 1x1x96x256) [24576, bytes: 98304]
InceptionResnetV1/Repeat/block35_2/Conv2d_1x1/biases:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Repeat/block35_3/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_3/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0c_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0c_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_3/Conv2d_1x1/weights:0 (float32_ref 1x1x96x256) [24576, bytes: 98304]
InceptionResnetV1/Repeat/block35_3/Conv2d_1x1/biases:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Repeat/block35_4/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_4/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0c_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0c_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_4/Conv2d_1x1/weights:0 (float32_ref 1x1x96x256) [24576, bytes: 98304]
InceptionResnetV1/Repeat/block35_4/Conv2d_1x1/biases:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Repeat/block35_5/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_5/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x32) [8192, bytes: 32768]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0c_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0c_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV1/Repeat/block35_5/Conv2d_1x1/weights:0 (float32_ref 1x1x96x256) [24576, bytes: 98304]
InceptionResnetV1/Repeat/block35_5/Conv2d_1x1/biases:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_6a/Branch_0/Conv2d_1a_3x3/weights:0 (float32_ref 3x3x256x384) [884736, bytes: 3538944]
InceptionResnetV1/Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm/beta:0 (float32_ref 384) [384, bytes: 1536]
InceptionResnetV1/Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm/moving_mean:0 (float32_ref 384) [384, bytes: 1536]
InceptionResnetV1/Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm/moving_variance:0 (float32_ref 384) [384, bytes: 1536]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x256x192) [49152, bytes: 196608]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x192x192) [331776, bytes: 1327104]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_1a_3x3/weights:0 (float32_ref 3x3x192x256) [442368, bytes: 1769472]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Repeat_1/block17_1/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_1/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_1/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_1/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_2/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_2/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_2/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_2/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_3/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_3/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_3/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_3/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_4/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_4/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_4/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_4/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_5/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_5/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_5/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_5/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_6/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_6/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_6/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_6/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_7/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_7/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_7/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_7/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_8/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_8/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_8/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_8/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_9/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_9/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_9/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_9/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Repeat_1/block17_10/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_10/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0b_1x7/weights:0 (float32_ref 1x7x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0b_1x7/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0c_7x1/weights:0 (float32_ref 7x1x128x128) [114688, bytes: 458752]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0c_7x1/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Repeat_1/block17_10/Conv2d_1x1/weights:0 (float32_ref 1x1x256x896) [229376, bytes: 917504]
InceptionResnetV1/Repeat_1/block17_10/Conv2d_1x1/biases:0 (float32_ref 896) [896, bytes: 3584]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x256) [229376, bytes: 917504]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_1a_3x3/weights:0 (float32_ref 3x3x256x384) [884736, bytes: 3538944]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/beta:0 (float32_ref 384) [384, bytes: 1536]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/moving_mean:0 (float32_ref 384) [384, bytes: 1536]
InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/moving_variance:0 (float32_ref 384) [384, bytes: 1536]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x256) [229376, bytes: 917504]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_1a_3x3/weights:0 (float32_ref 3x3x256x256) [589824, bytes: 2359296]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x896x256) [229376, bytes: 917504]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0b_3x3/weights:0 (float32_ref 3x3x256x256) [589824, bytes: 2359296]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0b_3x3/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_1a_3x3/weights:0 (float32_ref 3x3x256x256) [589824, bytes: 2359296]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_1a_3x3/BatchNorm/beta:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_1a_3x3/BatchNorm/moving_mean:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_1a_3x3/BatchNorm/moving_variance:0 (float32_ref 256) [256, bytes: 1024]
InceptionResnetV1/Repeat_2/block8_1/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_1/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0b_1x3/weights:0 (float32_ref 1x3x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0b_1x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0c_3x1/weights:0 (float32_ref 3x1x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0c_3x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_1/Conv2d_1x1/weights:0 (float32_ref 1x1x384x1792) [688128, bytes: 2752512]
InceptionResnetV1/Repeat_2/block8_1/Conv2d_1x1/biases:0 (float32_ref 1792) [1792, bytes: 7168]
InceptionResnetV1/Repeat_2/block8_2/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_2/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0b_1x3/weights:0 (float32_ref 1x3x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0b_1x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0c_3x1/weights:0 (float32_ref 3x1x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0c_3x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_2/Conv2d_1x1/weights:0 (float32_ref 1x1x384x1792) [688128, bytes: 2752512]
InceptionResnetV1/Repeat_2/block8_2/Conv2d_1x1/biases:0 (float32_ref 1792) [1792, bytes: 7168]
InceptionResnetV1/Repeat_2/block8_3/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_3/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0b_1x3/weights:0 (float32_ref 1x3x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0b_1x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0c_3x1/weights:0 (float32_ref 3x1x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0c_3x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_3/Conv2d_1x1/weights:0 (float32_ref 1x1x384x1792) [688128, bytes: 2752512]
InceptionResnetV1/Repeat_2/block8_3/Conv2d_1x1/biases:0 (float32_ref 1792) [1792, bytes: 7168]
InceptionResnetV1/Repeat_2/block8_4/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_4/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0b_1x3/weights:0 (float32_ref 1x3x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0b_1x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/weights:0 (float32_ref 3x1x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_4/Conv2d_1x1/weights:0 (float32_ref 1x1x384x1792) [688128, bytes: 2752512]
InceptionResnetV1/Repeat_2/block8_4/Conv2d_1x1/biases:0 (float32_ref 1792) [1792, bytes: 7168]
InceptionResnetV1/Repeat_2/block8_5/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_5/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0b_1x3/weights:0 (float32_ref 1x3x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0b_1x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0c_3x1/weights:0 (float32_ref 3x1x192x192) [110592, bytes: 442368]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0c_3x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Repeat_2/block8_5/Conv2d_1x1/weights:0 (float32_ref 1x1x384x1792) [688128, bytes: 2752512]
InceptionResnetV1/Repeat_2/block8_5/Conv2d_1x1/biases:0 (float32_ref 1792) [1792, bytes: 7168]
InceptionResnetV1/Block8/Branch_0/Conv2d_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Block8/Branch_0/Conv2d_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_0/Conv2d_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_0/Conv2d_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0a_1x1/weights:0 (float32_ref 1x1x1792x192) [344064, bytes: 1376256]
InceptionResnetV1/Block8/Branch_1/Conv2d_0a_1x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0b_1x3/weights:0 (float32_ref 1x3x192x192) [110592, bytes: 442368]
InceptionResnetV1/Block8/Branch_1/Conv2d_0b_1x3/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/weights:0 (float32_ref 3x1x192x192) [110592, bytes: 442368]
InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/BatchNorm/beta:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_mean:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance:0 (float32_ref 192) [192, bytes: 768]
InceptionResnetV1/Block8/Conv2d_1x1/weights:0 (float32_ref 1x1x384x1792) [688128, bytes: 2752512]
InceptionResnetV1/Block8/Conv2d_1x1/biases:0 (float32_ref 1792) [1792, bytes: 7168]
InceptionResnetV1/Bottleneck/weights:0 (float32_ref 1792x128) [229376, bytes: 917504]
InceptionResnetV1/Bottleneck/BatchNorm/beta:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Bottleneck/BatchNorm/moving_mean:0 (float32_ref 128) [128, bytes: 512]
InceptionResnetV1/Bottleneck/BatchNorm/moving_variance:0 (float32_ref 128) [128, bytes: 512]
Total size of variables: 22808144
Total bytes of variables: 91232576
'''



'''
res_regularization_loss:
 InceptionResnetV1/Conv2d_1a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Conv2d_2a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Conv2d_2b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Conv2d_3b_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Conv2d_4a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Conv2d_4b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_1/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_1/Branch_1/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_1/Branch_2/Conv2d_0c_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_1/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_2/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_2/Branch_1/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_2/Branch_2/Conv2d_0c_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_2/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_3/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_3/Branch_1/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_3/Branch_2/Conv2d_0c_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_3/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_4/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_4/Branch_1/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_4/Branch_2/Conv2d_0c_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_4/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_5/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_5/Branch_1/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_5/Branch_2/Conv2d_0c_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat/block35_5/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_6a/Branch_0/Conv2d_1a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_6a/Branch_1/Conv2d_1a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_1/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_1/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_1/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_2/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_2/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_3/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_3/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_3/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_4/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_4/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_4/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_5/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_5/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_5/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_6/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_6/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_6/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_7/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_7/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_7/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_8/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_8/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_8/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_9/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_9/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_9/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_10/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0b_1x7/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_10/Branch_1/Conv2d_0c_7x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_1/block17_10/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_7a/Branch_0/Conv2d_1a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_7a/Branch_1/Conv2d_1a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_0b_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Mixed_7a/Branch_2/Conv2d_1a_3x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_1/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0b_1x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_1/Branch_1/Conv2d_0c_3x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_1/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_2/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0b_1x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_2/Branch_1/Conv2d_0c_3x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_2/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_3/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0b_1x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_3/Branch_1/Conv2d_0c_3x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_3/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_4/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0b_1x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_4/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_5/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0b_1x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_5/Branch_1/Conv2d_0c_3x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Repeat_2/block8_5/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Block8/Branch_0/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Block8/Branch_1/Conv2d_0a_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Block8/Branch_1/Conv2d_0b_1x3/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Block8/Conv2d_1x1/kernel/Regularizer/l2_regularizer:0
res_regularization_loss:
 InceptionResnetV1/Bottleneck/kernel/Regularizer/l2_regularizer:0
'''