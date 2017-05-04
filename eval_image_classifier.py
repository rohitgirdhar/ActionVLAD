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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import dataset_data_provider

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'frames_per_video', 1, 'Number of frames per video.')

tf.app.flags.DEFINE_string(
    'gpus', '0', 'GPUs to use for testing.')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
if len(FLAGS.gpus.strip().split(',')) > 1:
  print('Multi-gpu testing not supported yet. Specify one gpu.')
  sys.exit(-1)


tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
# tf.app.flags.DEFINE_string(
#     'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'dataset_list_dir', '/home/rgirdhar/Work/Data/018_VideoVLAD/raw/UCF101/Lists/',
    'The directory where the dataset list files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_string(
    'bgr_flip', None, 'set true/false to turn on/off this, for each stream. Else use default.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_integer(
    'split_id', 1, 'Dataset split id to use.')

###########
# Pooling #
###########

tf.app.flags.DEFINE_string(
    'pooling', None,
    'Set =[netvlad/avg-conv] to train with that.')

tf.app.flags.DEFINE_string(
    'classifier_type', 'linear',
    'Classifier to use with netvlad/avg-conv. Use linear/two-layer.')

tf.app.flags.DEFINE_string('conv_endpoint', None,
                           'Set a non-default conv endpoint for netvlad.'
                           'Default for vgg16: conv5. Can set fc7.'
                           'Default for inceptionV2TSN is inception_5a.')

##############
# Store feat #
##############

tf.app.flags.DEFINE_string(
    'store_feat', None,
    'Set to comma sep list of endpoints to store.')

tf.app.flags.DEFINE_string(
    'store_feat_path', None,
    'Set to path of h5 file to write feats into.')

tf.app.flags.DEFINE_boolean(
    'force_random_shuffle', False,
    'Force random shuffle input data. Useful for storing training features for clustering.')

tf.app.flags.DEFINE_string('modality', 'rgb',
                           'Modality of training data.')

tf.app.flags.DEFINE_float('out_dim_scale', 1.0,
                          'Resize the output image by this scale. Eg, 224x '
                          'with 2 would be 448x images.')

tf.app.flags.DEFINE_integer('ncrops', 1,
                            'Number of image crops in testing. '
                            'Only 1 or 5 work.')

tf.app.flags.DEFINE_string('netvlad_initCenters', '',
                           'Path to PKL with the initial centers.')

tf.app.flags.DEFINE_string('stream_pool_type', None,
                           'Pool streams [concat-netvlad].')

tf.app.flags.DEFINE_integer('feat_store_compression_opt', 9,
                            'Compression opt for storing features.')



def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if not os.path.isfile(FLAGS.checkpoint_path):
    FLAGS.eval_dir = os.path.join(FLAGS.checkpoint_path, 'eval')
  else:
    FLAGS.eval_dir = os.path.join(
        os.path.dirname(FLAGS.checkpoint_path), 'eval')

  try:
    os.makedirs(FLAGS.eval_dir)
  except OSError:
    pass

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name,
        FLAGS.dataset_dir.split(','),
        FLAGS.dataset_list_dir,
        num_samples=FLAGS.frames_per_video,
        modality=FLAGS.modality,
        split_id=FLAGS.split_id)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        batch_size=FLAGS.batch_size,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=FLAGS.force_random_shuffle,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size,
        bgr_flips=FLAGS.bgr_flip)
    [image, label] = provider.get(['image', 'label'])
    label = tf.cast(tf.string_to_number(label, tf.int32),
        tf.int64)
    label.set_shape(())
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size,
                                   model_name=FLAGS.model_name,
                                   ncrops=FLAGS.ncrops,
                                   out_dim_scale=FLAGS.out_dim_scale)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=1 if FLAGS.store_feat is not None else FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    kwargs = {}
    if FLAGS.conv_endpoint is not None:
      kwargs['conv_endpoint'] = FLAGS.conv_endpoint
    logits, end_points = network_fn(
        images, pool_type=FLAGS.pooling,
        classifier_type=FLAGS.classifier_type,
        num_channels_stream=provider.num_channels_stream,
        netvlad_centers=FLAGS.netvlad_initCenters.split(','),
        stream_pool_type=FLAGS.stream_pool_type,
        **kwargs)
    end_points['images'] = images
    end_points['labels'] = labels

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    # rgirdhar: Because of the following, can't use with batch_size=1
    if FLAGS.batch_size > 1:
      labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall@5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.iteritems():
      summary_name = 'eval/%s' % name
      op = tf.scalar_summary(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = int(math.ceil(dataset.num_samples /
                                  float(FLAGS.batch_size)))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if FLAGS.store_feat is not None:
      assert(FLAGS.store_feat_path is not None)
      from tensorflow.python.training import supervisor
      from tensorflow.python.framework import ops
      import h5py
      saver = tf.train.Saver(variables_to_restore)
      sv = supervisor.Supervisor(graph=ops.get_default_graph(),
                                 logdir=None,
                                 summary_op=None,
                                 summary_writer=None,
                                 global_step=None,
                                 saver=None)
      ept_names_to_store = FLAGS.store_feat.split(',')
      try:
        ept_to_store = [end_points[el] for el in ept_names_to_store]
      except:
        logging.error('Endpoint not found')
        logging.error('Choose from %s' % ','.join(end_points.keys()))
        raise KeyError()
      res = dict([(epname, []) for epname in ept_names_to_store])
      with sv.managed_session(
          FLAGS.master, start_standard_services=False,
          config=config) as sess:
        saver.restore(sess, checkpoint_path)
        sv.start_queue_runners(sess)
        for j in range(num_batches):
          if j % 10 == 0:
            logging.info('Doing batch %d/%d' % (j, num_batches))
          feats = sess.run(ept_to_store)
          for eid, epname in enumerate(ept_names_to_store):
            res[epname].append(feats[eid])
      logging.info('Writing out features to %s' % FLAGS.store_feat_path)
      with h5py.File(FLAGS.store_feat_path, 'w') as fout:
        for epname in res.keys():
          fout.create_dataset(epname,
              data=np.concatenate(res[epname], axis=0),
              compression='gzip',
              compression_opts=FLAGS.feat_store_compression_opt)
    else:
      slim.evaluation.evaluate_once(
          master=FLAGS.master,
          checkpoint_path=checkpoint_path,
          logdir=FLAGS.eval_dir,
          num_evals=num_batches,
          eval_op=names_to_updates.values(),
          variables_to_restore=variables_to_restore,
          session_config=config)


if __name__ == '__main__':
  tf.app.run()
