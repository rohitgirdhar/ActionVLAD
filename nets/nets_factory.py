# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
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
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from nets import vgg
from nets import frame_pooling as pooling
from nets import inception

slim = tf.contrib.slim

networks_map = {'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'inception_v2_tsn': inception.inception_v2_tsn,
               }

arg_scopes_map = {'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'inception_v2_tsn': inception.inception_v2_tsn_arg_scope,
                 }

def split_images(images, num_channels_stream):
  if num_channels_stream is None:
    return [images]
  images_splits = []
  cur_pos = 0
  for pos in num_channels_stream:
    images_splits.append(images[..., cur_pos : cur_pos+pos])
    cur_pos += pos
  return images_splits


def get_network_fn(
    name, num_classes, batch_size,
    weight_decay=0.0, is_training=False,
    dropout_keep_prob=0.2,
    pooled_dropout_keep_prob=0.5,
    batch_norm=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images, pool_type=None,
                 classifier_type=None,
                 num_channels_stream=None,
                 netvlad_centers=[],
                 stream_pool_type=None,
                 **kwargs):
    num_image_sets = 1
    if len(images.get_shape()) == 5:
      num_image_sets = images.get_shape().as_list()[0]
      images = tf.reshape(images, [-1, ] + images.get_shape().as_list()[2:])
    images_sets = split_images(images, num_channels_stream)
    all_end_points = []
    all_out_nets = []
    with slim.arg_scope(arg_scope):
      for sid,images in enumerate(images_sets):
        with tf.variable_scope('stream%d' % sid):
          net, end_points = func(images,
                                 num_classes,
                                 is_training=is_training,
                                 dropout_keep_prob=dropout_keep_prob,
                                 conv_only=(pool_type == 'netvlad' or
                                  pool_type == 'avg-conv' or
                                  pool_type == 'max-conv' or
                                  stream_pool_type ==
                                            'concat-last-conv-and-netvlad' or
                                  stream_pool_type == 'one-bag-and-netvlad'),
                                 **kwargs)
          all_out_nets.append(net)
          if pool_type in ['netvlad', 'avg-conv', 'max-conv']:
            # last_conv = end_points[tf.get_variable_scope().name + '/' +
            #                       conv_endpoint_map[name]]
            last_conv = net  # both VGG and resnet have conv_only implemented
            if pool_type == 'netvlad':
              net, netvlad_end_points = \
                  pooling.netvlad(last_conv, batch_size, 0.0,
                                  netvlad_initCenters=netvlad_centers[sid])
              end_points[tf.get_variable_scope().name + '/netvlad'] = net
              end_points.update(netvlad_end_points)
            elif pool_type == 'avg-conv':
              net = pooling.pool_conv(last_conv, batch_size, 'avg')
              end_points[tf.get_variable_scope().name + '/avg-conv'] = net
            elif pool_type == 'max-conv':
              net = pooling.pool_conv(last_conv, batch_size, 'max')
              end_points[tf.get_variable_scope().name + '/max-conv'] = net
            if batch_norm:
              with tf.variable_scope('pooled-batch-norm'):
                net = slim.batch_norm(net, is_training=is_training)

            if classifier_type is not None and classifier_type != 'None':
              print('Dropout is not being applied to the model to be consistent with original release of the code. '
                    'Due to an issue it was not enabled in the original release. '
                    'Please uncomment lines in nets/nets_factory.py to enable the dropout.')
              # net = slim.dropout(net, pooled_dropout_keep_prob, scope='pooled-dropout',
              #                    is_training=is_training)
            if classifier_type == 'linear':
              with tf.variable_scope('classifier'):
                net = slim.fully_connected(
                    net, num_classes,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logits')
            elif classifier_type == 'two-layer':
              with tf.variable_scope('classifier'):
                net = slim.fully_connected(
                    net, 4096,
                    scope='logits-1')
                net = slim.fully_connected(
                    net, num_classes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logits-2')
            end_points[tf.get_variable_scope().name + '/logits'] = net
          elif pool_type in ['avg', 'avg-after-softmax']:
            if pool_type == 'avg-after-softmax':
              net = tf.nn.softmax(net)
            video_frames = tf.split(0, num_image_sets, net)
            net = tf.concat(0, [tf.reduce_mean(el, 0, keep_dims=True)
                                   for el in video_frames])
            end_points[tf.get_variable_scope().name + '/logits'] = net
          all_end_points.append(end_points)
      all_end_points.append({})  # for the stream concat ops
      if stream_pool_type == 'concat-netvlad':
        assert(len(all_end_points) == 2+1) # TODO: fix this
        net = tf.concat(1,
                        (all_end_points[0]['stream0/netvlad'],
                         all_end_points[1]['stream1/netvlad']))
        all_end_points[-1]['concat-netvlad'] = net
      elif stream_pool_type == 'wtd-avg-pool-logits':
        assert(len(all_end_points) == 2+1) # TODO: fix this
        net = 0.667 * all_end_points[1]['stream1/logits'] + \
              0.333 * all_end_points[0]['stream0/logits']
        all_end_points[-1]['wtd-avg-pool-logits'] = net
      elif stream_pool_type == 'concat-last-conv-and-netvlad' or \
           stream_pool_type == 'one-bag-and-netvlad':
        with tf.variable_scope(stream_pool_type):
          if stream_pool_type == 'one-bag-and-netvlad':
            net = tf.concat(1, all_out_nets)
          else:
            net = tf.concat(3, all_out_nets)
          end_points[tf.get_variable_scope().name + '/concat-last-conv'] = net
          net, netvlad_end_points = pooling.netvlad(
            net, batch_size, 0.0, netvlad_initCenters=netvlad_centers[0])
          end_points[tf.get_variable_scope().name + '/concat-last-conv-netvlad'] = net
          end_points.update(netvlad_end_points)
      elif stream_pool_type is not None:
        raise ValueError('Unknown stream pool type %s' % stream_pool_type)
      if stream_pool_type in ['concat-netvlad', 'concat-last-conv-and-netvlad',
                              'one-bag-and-netvlad']:
        net = slim.dropout(net, pooled_dropout_keep_prob, scope='pooled-dropout',
                           is_training=is_training)
        with tf.variable_scope('classifier'):
          net = slim.fully_connected(
            net, num_classes,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            activation_fn=None,
            normalizer_fn=None,
            scope='stream-pool-logits')
        all_end_points[-1]['stream_pool_type'] = net
      logits = net

    final_end_points = {}
    for el in all_end_points:
      final_end_points.update(el)

    return logits, final_end_points
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
