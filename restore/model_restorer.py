# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import numpy as np
import h5py

from tensorflow.contrib import slim
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
import var_name_mapper


def restore_model(checkpoint_paths,
                  variables_to_restore,
                  ignore_missing_vars=False,
                  num_streams=1,
                  checkpoint_style=None,
                  special_assign_vars=None):
    all_ops = []
    if len(checkpoint_paths) == 1 and num_streams > 1:
      logging.info('Provided one checkpoint for multi-stream '
                   'network. Will use this as a saved model '
                   'with this exact multi stream network.')
      all_ops.append(slim.assign_from_checkpoint_fn(
        checkpoint_paths[0],
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars))
    else:
      for sid in range(num_streams):
        this_checkpoint_style = checkpoint_style.split(',')[sid] if \
                                checkpoint_style is not None else None
        checkpoint_path = checkpoint_paths[sid]
        # assert tf.gfile.Exists(checkpoint_path)
        this_stream_name = 'stream%d/' % sid
        this_checkpoint_variables = [var for var in variables_to_restore
                                     if var in
                                     slim.get_model_variables(this_stream_name)]
        if checkpoint_path.endswith('.npy'):
          vars_to_restore_names = [
              el.name for el in this_checkpoint_variables]
          key_name_mapper = var_name_mapper.map()
          init_weights = np.load(checkpoint_path).item()
          init_weights_final = {}
          vars_restored = []
          for key in init_weights.keys():
            for subkey in init_weights[key].keys():
              prefix = this_stream_name
              if this_checkpoint_style == 'v2_withStream':
                prefix = 'stream0/'  # because any model trained with stream
                                     # will have that stream as 0
              final_key_name = prefix + key_name_mapper(
                  key + '/' + subkey)
              if final_key_name not in vars_to_restore_names:
                logging.error('Not using %s from npy' % final_key_name)
                continue
              
              target_shape = slim.get_model_variables(
                final_key_name)[0].get_shape().as_list()
              pretrained_wts = init_weights[key][subkey]
              target_shape_squeezed = np.delete(
                target_shape, np.where(np.array(target_shape) == 1))
              pretrained_shape_squeezed = np.delete(
                pretrained_wts.shape, np.where(np.array(pretrained_wts.shape) == 1))
              if np.all(target_shape_squeezed !=
                        pretrained_shape_squeezed):
                logging.error('Shape mismatch var: %s from npy [%s vs %s]' 
                              % (final_key_name, target_shape,
                                 pretrained_wts.shape))

              init_weights_final[final_key_name] = \
                  pretrained_wts
              vars_restored.append(final_key_name)
          init_weights = init_weights_final
          for v in vars_to_restore_names:
            if v not in vars_restored:
              logging.fatal('No weights found for %s' % v)
          all_ops.append(slim.assign_from_values_fn(
              init_weights))
        else:
          if this_checkpoint_style != 'v2_withStream':
            all_ops.append(slim.assign_from_checkpoint_fn(
                checkpoint_path,
                # stripping the stream name to map variables
                dict(
                  [('/'.join(el.name.split('/')[1:]).split(':')[0], el) for
                      el in this_checkpoint_variables]),
                ignore_missing_vars=ignore_missing_vars))
          else:
            all_ops.append(slim.assign_from_checkpoint_fn(
                checkpoint_path,
                # stripping the stream name to map variables, to stream0,
                # as the model is v2_withStream, hence must be trained with
                # stream0/ prefix
                dict(
                  [('/'.join(['stream0'] + el.name.split('/')[1:]).split(':')[0], el) for
                      el in this_checkpoint_variables]),
                ignore_missing_vars=ignore_missing_vars))
    if special_assign_vars is not None:
      all_ops.append(get_special_assigns(special_assign_vars))
    def combined(sess):
      for op in all_ops:
        op(sess)
    return combined


def get_special_assigns(special_assign_vars):
  init_wts = {}
  special_assign_vars = special_assign_vars.split(',')
  for i in range(len(special_assign_vars) / 2):
    var_name = special_assign_vars[2*i]
    file_path = special_assign_vars[2*i+1]
    with h5py.File(file_path, 'r') as fin:
      init_wts[var_name] = fin['feat'].value
    logging.info('Special Assign: %s with a %s array' % (
      var_name, init_wts[var_name].shape))
  return slim.assign_from_values_fn(init_wts)
