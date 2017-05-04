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

import tensorflow as tf

from preprocessing import vgg_ucf_preprocessing

slim = tf.contrib.slim


def get_preprocessing(name, is_training=False):
  """Returns preprocessing_fn(image, height, width, **kwargs).

  Args:
    name: The name of the preprocessing function.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    preprocessing_fn: A function that preprocessing a single image (pre-batch).
      It has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).

  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """
  preprocessing_fn_map = {
      'vgg_ucf': vgg_ucf_preprocessing,
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, output_height, output_width, **kwargs):
    with tf.variable_scope('preprocess_image'):
      if len(image.get_shape()) == 3:
        return preprocessing_fn_map[name].preprocess_image(
            image, output_height, output_width, is_training=is_training, **kwargs)
      elif len(image.get_shape()) == 4:
        # preprocess all the images in one set in the same way by concat-ing
        # them in channels
        nImgs = image.get_shape().as_list()[0]
        final_img_concat = preprocessing_fn_map[name].preprocess_image(
            tf.concat(2, tf.unpack(image)),
            output_height, output_width, is_training=is_training, **kwargs)
        return tf.concat(0, tf.split(3, nImgs, final_img_concat))
      else:
        print('Incorrect dims image!')

  return preprocessing_fn
