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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.client import timeline
from datasets import dataset_factory
from datasets import dataset_data_provider
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from restore import model_restorer

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string('gpus', '0',
                           'Comma sep list of gpus.')

tf.app.flags.DEFINE_integer('num_clones', -1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'cpu_threads', 0,
    'Max CPU threads used apart from data prefetch. '
    '0 default lets system pick a reasonable number.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 30,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0e-8, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

tf.app.flags.DEFINE_float(
    'clip_gradients', 0,
    'Clip gradients by this value.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor',
    # 0.94,
    0.1,
    'Learning rate decay factor.')

# tf.app.flags.DEFINE_float(
#     'num_epochs_per_decay', 2.0,
#     'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_integer(
    'num_steps_per_decay', 10000,
    'Number of steps after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

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
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_float(
    'dropout', 0.8, 'Dropout on last layers.')

tf.app.flags.DEFINE_float(
    'pooled_dropout', 0.5, 'Dropout on conv-pooled/netvlad output.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_string(
    'bgr_flip', None,
    ('Set true or false to force either, for each stream. As none (default) it will do'
      'whatever is default for that preprocessor'))

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'frames_per_video', 1, 'The number of frames in each batch element.')

tf.app.flags.DEFINE_integer(
    'iter_size', 1, 'Number of forward iterations before a back.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string('modality', 'rgb',
                           'Modality of training data.')

tf.app.flags.DEFINE_string('scale_ratios', '1,0.875,0.75,0.66',
                           'Ratios to scale the frames by for augmentation.')

tf.app.flags.DEFINE_float('out_dim_scale', 1.0,
                          'Resize the output image by this scale. Eg, 224x '
                          'with 2 would be 448x images.')

tf.app.flags.DEFINE_integer('split_id', 1,
                            'Dataset split to use.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_style', None,
    'Comma separated list of type of each checkpoint. [v1/v2_withStream]. '
    'Default all are v1.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_boolean(
    'debug', False,
    'Running some debugging print ops.')

tf.app.flags.DEFINE_string(
    'special_assign_vars', None,
    'Specify some variables to assigned using specific files. '
    'Use format: <var_name1>,<file_name1>,<var_name2>,<file_name2>...')

###########
# NetVLAD #
###########

tf.app.flags.DEFINE_string(
    'pooling', None,
    'Set =[netvlad/avg-conv] to train with that.')

tf.app.flags.DEFINE_string(
    'classifier_type', 'linear',
    'Classifier to use with netvlad/avg-conv. Use linear/two-layer.')

tf.app.flags.DEFINE_boolean(
    'netvlad_batch_norm', False,
    'Apply a batch norm to the netvlad features.')

tf.app.flags.DEFINE_integer(
    'num_streams', 1,
    'Number of [flow/rgb etc] streams.')

tf.app.flags.DEFINE_string('netvlad_initCenters', '',
                           'Path to PKL with the initial centers.')

tf.app.flags.DEFINE_string('stream_pool_type', None,
                           'Pool streams [concat-netvlad].')

tf.app.flags.DEFINE_string('conv_endpoint', None,
                           'Set a non-default conv endpoint for netvlad.'
                           'Default for vgg16: conv5. Can set fc7.'
                           'Default for inceptionV2TSN is inception_5a.')

#########
# Other #
#########

tf.app.flags.DEFINE_boolean(
    'profile_iterations', False,
    'Write timeline profile of each iteration.')


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  # decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
  #                   FLAGS.num_epochs_per_decay)
  decay_steps = FLAGS.num_steps_per_decay
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _add_variables_summaries(learning_rate):
  summaries = []
  for variable in slim.get_model_variables():
    summaries.append(tf.histogram_summary(variable.op.name, variable))
  summaries.append(tf.scalar_summary('training/Learning Rate', learning_rate))
  return summaries


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  checkpoint_paths = FLAGS.checkpoint_path.split(',')
  for cid in range(len(checkpoint_paths)):
    if tf.gfile.IsDirectory(checkpoint_paths[cid]):
      checkpoint_paths[cid] = tf.train.latest_checkpoint(checkpoint_paths[cid])

  tf.logging.info('Fine-tuning from %s' % FLAGS.checkpoint_path)

  return model_restorer.restore_model(
      checkpoint_paths,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars,
      num_streams=FLAGS.num_streams,
      checkpoint_style=FLAGS.checkpoint_style,
      special_assign_vars=FLAGS.special_assign_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


end_points_debug = []
def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
  Args:
    sess: The current session.
    train_op: A dictionary of `Operation` that evaluates the gradients and returns the
      total loss (for first) in case of iter_size > 1.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.
  Returns:
    The total loss and a boolean indicating whether or not to stop training.
  """
  start_time = time.time()
  if FLAGS.iter_size == 1:
    # for debugging specific endpoint values,
    # set the train file to one image and use
    # pdb here
    # import pdb
    # pdb.set_trace()
    if FLAGS.profile_iterations:
      run_options = tf.RunOptions(
          trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      total_loss, np_global_step = sess.run([train_op, global_step],
          options=run_options,
          run_metadata=run_metadata)
      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open(os.path.join(FLAGS.train_dir,
                             'timeline_%08d.json' % np_global_step), 'w') as f:
        f.write(ctf)
    else:
      total_loss, np_global_step = sess.run([train_op, global_step])
  else:
    for j in range(FLAGS.iter_size-1):
      sess.run([train_op[j]])
    total_loss, np_global_step = sess.run(
        [train_op[FLAGS.iter_size-1], global_step])
  time_elapsed = time.time() - start_time

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('%s: global step %d: loss = %.4f (%.2f sec)',
                   datetime.now(), np_global_step, total_loss, time_elapsed)

  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop


def summarize_images(images, num_channels_stream):
  from nets.nets_factory import split_images
  images_list = split_images(images, num_channels_stream)
  for sid,im_st in enumerate(images_list):
    ndims = im_st.get_shape().ndims
    if im_st.get_shape().as_list()[-1] != 3:
      # im_st = tf.expand_dims(tf.transpose(im_st, [ndims-1] + range(ndims-1)),
      #                       ndims)
      im_st = tf.reduce_mean(im_st, reduction_indices=ndims-1,
                             keep_dims=True)
    if ndims > 4:
      im_st = tf.reshape(im_st, [-1,] + im_st.get_shape().as_list()[-3:])
    tf.image_summary('stream%d' % sid, im_st / 128)


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  if FLAGS.num_clones == -1:
    FLAGS.num_clones = len(FLAGS.gpus.split(','))

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # tf.set_random_seed(42)
    tf.set_random_seed(0)
    ######################
    # Config model_deploy#
    ######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name,
        FLAGS.dataset_dir.split(','),
        dataset_list_dir=FLAGS.dataset_list_dir,
        num_samples=FLAGS.frames_per_video,
        modality=FLAGS.modality,
        split_id=FLAGS.split_id)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        batch_size=FLAGS.batch_size,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        dropout_keep_prob=(1.0-FLAGS.dropout),
        pooled_dropout_keep_prob=(1.0-FLAGS.pooled_dropout),
        batch_norm=FLAGS.netvlad_batch_norm)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)  # in case of pooling images,
                           # now preprocessing is done video-level

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size,
        bgr_flips=FLAGS.bgr_flip)
      [image, label] = provider.get(['image', 'label'])
      # now note that the above image might be a 23 channel image if you have
      # both RGB and flow streams. It will need to split later, but all the
      # preprocessing will be done consistently for all frames over all streams
      label = tf.string_to_number(label, tf.int32)
      label.set_shape(())
      label -= FLAGS.labels_offset

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      scale_ratios=[float(el) for el in FLAGS.scale_ratios.split(',')],
      image = image_preprocessing_fn(image, train_image_size,
                                     train_image_size,
                                     scale_ratios=scale_ratios,
                                     out_dim_scale=FLAGS.out_dim_scale,
                                     model_name=FLAGS.model_name)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      if FLAGS.debug:
        images = tf.Print(images, [labels], 'Read batch')
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)
      summarize_images(images, provider.num_channels_stream)

    ####################
    # Define the model #
    ####################
    kwargs = {}
    if FLAGS.conv_endpoint is not None:
      kwargs['conv_endpoint'] = FLAGS.conv_endpoint
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(
          images, pool_type=FLAGS.pooling,
          classifier_type=FLAGS.classifier_type,
          num_channels_stream=provider.num_channels_stream,
          netvlad_centers=FLAGS.netvlad_initCenters.split(','),
          stream_pool_type=FLAGS.stream_pool_type,
          **kwargs)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weight=0.4, scope='aux_loss')
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weight=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    global end_points_debug
    end_points = clones[0].outputs
    end_points_debug = dict(end_points)
    end_points_debug['images'] = images
    end_points_debug['labels'] = labels
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.histogram_summary('activations/' + end_point, x))
      summaries.add(tf.scalar_summary('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.scalar_summary('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.histogram_summary(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.scalar_summary('learning_rate', learning_rate,
                                      name='learning_rate'))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()
    logging.info('Training the following variables: %s' % (
      ' '.join([el.name for el in variables_to_train])))

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)

    # clip the gradients if needed
    if FLAGS.clip_gradients > 0:
      logging.info('Clipping gradients by %f' % FLAGS.clip_gradients)
      with tf.name_scope('clip_gradients'):
        clones_gradients = slim.learning.clip_gradient_norms(
            clones_gradients,
            FLAGS.clip_gradients)

    # Add total_loss to summary.
    summaries.add(tf.scalar_summary('total_loss', total_loss,
                                    name='total_loss'))

    # Create gradient updates.
    train_ops = {}
    if FLAGS.iter_size == 1:
      grad_updates = optimizer.apply_gradients(clones_gradients,
                                               global_step=global_step)
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                        name='train_op')
      train_ops = train_tensor
    else:
      gvs = [(grad, var) for grad, var in clones_gradients]
      varnames = [var.name for grad, var in gvs]
      varname_to_var = {var.name: var for grad, var in gvs}
      varname_to_grad = {var.name: grad for grad, var in gvs}
      varname_to_ref_grad = {}
      for vn in varnames:
        grad = varname_to_grad[vn]
        print("accumulating ... ", (vn, grad.get_shape()))
        with tf.variable_scope("ref_grad"):
          with tf.device(deploy_config.variables_device()):
            ref_var = slim.local_variable(
                np.zeros(grad.get_shape(),dtype=np.float32),
                name=vn[:-2])
            varname_to_ref_grad[vn] = ref_var

      all_assign_ref_op = [ref.assign(varname_to_grad[vn]) for vn, ref in varname_to_ref_grad.items()]
      all_assign_add_ref_op = [ref.assign_add(varname_to_grad[vn]) for vn, ref in varname_to_ref_grad.items()]
      assign_gradients_ref_op = tf.group(*all_assign_ref_op)
      accmulate_gradients_op = tf.group(*all_assign_add_ref_op)
      with tf.control_dependencies([accmulate_gradients_op]):
        final_gvs = [(varname_to_ref_grad[var.name] / float(FLAGS.iter_size), var) for grad, var in gvs]
        apply_gradients_op = optimizer.apply_gradients(final_gvs, global_step=global_step)
        update_ops.append(apply_gradients_op)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op],
            total_loss, name='train_op')
      for i in range(FLAGS.iter_size):
        if i == 0:
          train_ops[i] = assign_gradients_ref_op
        elif i < FLAGS.iter_size - 1:  # because apply_gradients also computes
                                       # (see control_dependency), so
                                       # no need of running an extra iteration
          train_ops[i] = accmulate_gradients_op
        else:
          train_ops[i] = train_tensor


    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.merge_summary(list(summaries), name='summary_op')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = FLAGS.cpu_threads
    # config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction=0.7

    ###########################
    # Kicks off the training. #
    ###########################
    logging.info('RUNNING ON SPLIT %d' % FLAGS.split_id)
    slim.learning.train(
        train_ops,
        train_step_fn=train_step,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None,
        session_config=config)


if __name__ == '__main__':
  tf.app.run()
