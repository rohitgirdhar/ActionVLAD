# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import numpy as np
import cPickle as pickle

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.platform import tf_logging as logging

FLAGS = tf.app.flags.FLAGS
# NetVLAD Parameters
tf.app.flags.DEFINE_float('netvlad_alpha', 1000.0,
                          """Alpha to use for netVLAD.""")


def softmax(target, axis, name=None):
    with tf.name_scope(name, 'softmax', [target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax


def netvlad(net, videos_per_batch, weight_decay, netvlad_initCenters):
    end_points = {}
    # VLAD pooling
    try:
      netvlad_initCenters = int(netvlad_initCenters)
      # initialize the cluster centers randomly
      cluster_centers = np.random.normal(size=(
        netvlad_initCenters, net.get_shape().as_list()[-1]))
      logging.info('Randomly initializing the {} netvlad cluster '
                   'centers'.format(cluster_centers.shape))
    except ValueError:
      with open(netvlad_initCenters, 'rb') as fin:
        kmeans = pickle.load(fin)
        cluster_centers = kmeans.cluster_centers_
    with tf.variable_scope('NetVLAD'):
        # normalize features
        net_normed = tf.nn.l2_normalize(net, 3, name='FeatureNorm')
        end_points[tf.get_variable_scope().name + '/net_normed'] = net_normed
        vlad_centers = slim.model_variable(
            'centers',
            shape=cluster_centers.shape,
            initializer=tf.constant_initializer(cluster_centers),
            regularizer=slim.l2_regularizer(weight_decay))
        end_points[tf.get_variable_scope().name + '/vlad_centers'] = vlad_centers
        vlad_W = slim.model_variable(
            'vlad_W',
            shape=(1, 1, ) + cluster_centers.transpose().shape,
            initializer=tf.constant_initializer(
                cluster_centers.transpose()[np.newaxis, np.newaxis, ...] *
                2 * FLAGS.netvlad_alpha),
            regularizer=slim.l2_regularizer(weight_decay))
        end_points[tf.get_variable_scope().name + '/vlad_W'] = vlad_W
        vlad_B = slim.model_variable(
            'vlad_B',
            shape=cluster_centers.shape[0],
            initializer=tf.constant_initializer(
                -FLAGS.netvlad_alpha *
                np.sum(np.square(cluster_centers), axis=1)),
            regularizer=slim.l2_regularizer(weight_decay))
        end_points[tf.get_variable_scope().name + '/vlad_B'] = vlad_B
        conv_output = tf.nn.conv2d(net_normed, vlad_W, [1, 1, 1, 1], 'VALID')
        dists = tf.nn.bias_add(conv_output, vlad_B)
        assgn = softmax(dists, axis=3)
        end_points[tf.get_variable_scope().name + '/assgn'] = assgn

        vid_splits = tf.split(0, videos_per_batch, net_normed)
        assgn_splits = tf.split(0, videos_per_batch, assgn)
        num_vlad_centers = vlad_centers.get_shape()[0]
        vlad_centers_split = tf.split(0, num_vlad_centers, vlad_centers)
        final_vlad = []
        for feats, assgn in zip(vid_splits, assgn_splits):
            vlad_vectors = []
            assgn_split_byCluster = tf.split(3, num_vlad_centers, assgn)
            for k in range(num_vlad_centers):
                res = tf.reduce_sum(
                    tf.mul(tf.sub(
                    feats,
                    vlad_centers_split[k]), assgn_split_byCluster[k]),
                    [0, 1, 2])
                vlad_vectors.append(res)
            vlad_vectors_frame = tf.pack(vlad_vectors, axis=0)
            final_vlad.append(vlad_vectors_frame)
        vlad_rep = tf.pack(final_vlad, axis=0, name='unnormed-vlad')
        end_points[tf.get_variable_scope().name + '/unnormed_vlad'] = vlad_rep
        with tf.name_scope('intranorm'):
            intranormed = tf.nn.l2_normalize(vlad_rep, dim=2)
        end_points[tf.get_variable_scope().name + '/intranormed_vlad'] = intranormed
        with tf.name_scope('finalnorm'):
            vlad_rep = tf.nn.l2_normalize(tf.reshape(
                intranormed,
                [intranormed.get_shape().as_list()[0], -1]),
                dim=1)
    return vlad_rep, end_points


def pool_conv(net, videos_per_batch, type='avg'):
    """
    Pool all the features across the frame and across all the frames
    for the video to get a single representation.
    Useful as a way to debug NetVLAD, as this should be worse than 
    NetVLAD with k = 1.
    """
    if type == 'avg':
      method = tf.reduce_mean
    elif type == 'max':
      method = tf.reduce_max
    else:
      raise ValueError('Not Found')
    with tf.name_scope('%s-conv' % type):
        vid_splits = tf.split(0, videos_per_batch, net);
        vids_pooled = [method(vid, [0, 1, 2]) for vid in vid_splits]
        return tf.pack(vids_pooled, axis=0)
