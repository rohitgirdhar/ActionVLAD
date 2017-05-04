# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from preprocessing.utils import _mean_image_subtraction
from preprocessing.vgg_preprocessing import _crop, _central_crop
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

_R_MEAN = 123.0
_G_MEAN = 117.0
_B_MEAN = 104.0

_RESIZE_HT = 256
_RESIZE_WD = 340  # This used to be 340 in the previous code
_SCALE_RATIOS = [1,.875,.75,.66]


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  image_shape = control_flow_ops.with_dependencies(
      [rank_assertions[0]],
      tf.shape(image_list[0]))
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                               tf.shape(image))
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  # TODO (rgirdhar): Force corner crops, right now going with random crops
  max_offset_height = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         mean_vals,
                         out_dim_scale=1.0):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
  num_channels = image.get_shape().as_list()[-1]
  image = tf.image.resize_images(image, [_RESIZE_HT, _RESIZE_WD])
  # compute the crop size
  base_size = float(min(_RESIZE_HT, _RESIZE_WD))
  scale_ratio_h = tf.random_shuffle(tf.constant(_SCALE_RATIOS))[0]
  scale_ratio_w = tf.random_shuffle(tf.constant(_SCALE_RATIOS))[0]
  image = _random_crop([image],
      tf.cast(output_height * scale_ratio_h, tf.int32),
      tf.cast(output_width * scale_ratio_w, tf.int32))[0]
  image = tf.image.resize_images(
    image, [int(output_height * out_dim_scale),
            int(output_width * out_dim_scale)])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)
  image.set_shape([int(output_height * out_dim_scale),
                   int(output_width * out_dim_scale), num_channels])
  image = _mean_image_subtraction(image, mean_vals)
  image = tf.expand_dims(image, 0) # 1x... image, to be consistent with eval
  # Gets logged multiple times with NetVLAD, so gives an error.
  # I'm anyway logging from the train code, so removing it here.
  # tf.image_summary('final_distorted_image',
  #     tf.expand_dims(image / 128.0, 0))
  return image


def preprocess_for_eval(image, output_height, output_width,
                        mean_vals, out_dim_scale=1.0, ncrops=1):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
  image = tf.image.resize_images(image, [_RESIZE_HT, _RESIZE_WD])
  if ncrops == 1:
    images = tf.image.crop_to_bounding_box(image, 16, 60, output_height, output_width)
    if abs(out_dim_scale - 1) >= 0.1:
      images = tf.image.resize_images(
        images, [int(output_height * out_dim_scale),
                int(output_width * out_dim_scale)])
    images = _mean_image_subtraction(tf.to_float(images), mean_vals)
    images = tf.expand_dims(images, 0)
  elif ncrops == 5:
    collect = []
    collect.append(tf.image.crop_to_bounding_box(
      image, 0, 0, output_height, output_width))
    collect.append(tf.image.crop_to_bounding_box(
      image, 0, 115, output_height, output_width))
    collect.append(tf.image.crop_to_bounding_box(
      image, 31, 0, output_height, output_width))
    collect.append(tf.image.crop_to_bounding_box(
      image, 31, 115, output_height, output_width))
    collect.append(tf.image.crop_to_bounding_box(
      image, 16, 60, output_height, output_width))
    # for i in range(5):
    #   collect.append(tf.image.flip_left_right(collect[i]))
    for i in range(len(collect)):
      collect[i] = _mean_image_subtraction(tf.to_float(collect[i]), mean_vals)
    images = tf.pack(collect)
  # image.set_shape([output_height, output_width, 3])
  # images = tf.to_float(images)
  return images


def preprocess_image(image, output_height, output_width, is_training=False,
                     scale_ratios=_SCALE_RATIOS,
                     ncrops=1,
                     out_dim_scale=1.0,  # scale the output image dimensions by
                                         # this ratio
                     model_name='vgg_16'):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  global _SCALE_RATIOS
  _SCALE_RATIOS = scale_ratios[0]
  # my image_read_utils normalizes the image to 0-1, so restore that
  IMG_SCALER = 255.0
  FLOW_MEAN = 128
  FINAL_SCALER = 1.0
  global _B_MEAN, _G_MEAN, _R_MEAN
  if model_name.startswith('inception') and model_name != 'inception_v2_tsn':
    logging.info('Using inception parameters for preprocessing')
    IMG_SCALER = 1.0
    _B_MEAN = 0.5
    _R_MEAN = 0.5
    _G_MEAN = 0.5
    FLOW_MEAN = 0.5
    FINAL_SCALER = 2.0
  image = image * IMG_SCALER
  num_channels = image.get_shape().as_list()[-1]
  if num_channels % 3 == 0:
    # For RGB, anyway it's always BGR flipped
    # if bgr_flip:
    logging.info('Assuming the batch is full of RGB images, and BGR flipped')
    mean_vals = [_B_MEAN, _G_MEAN, _R_MEAN]
    # else:
    #   mean_vals = [_R_MEAN, _G_MEAN, _B_MEAN]
  elif num_channels % 23 == 0:
    logging.info('Assuming the batch is full of RGB+Flow images, and first '
                 'part is BGR flipped')
    mean_vals = [_B_MEAN, _G_MEAN, _R_MEAN] + [FLOW_MEAN] * 20
  else:
    logging.info('Assuming the batch is full of Flow images.')
    mean_vals = [FLOW_MEAN] * num_channels
  if is_training:
    res = preprocess_for_train(image, output_height, output_width, mean_vals,
                               out_dim_scale)
  else:
    logging.info('Performing eval pre-processing')
    res = preprocess_for_eval(image, output_height, output_width, mean_vals,
                              out_dim_scale, ncrops)
  return res * FINAL_SCALER
