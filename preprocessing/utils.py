# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import tensorflow as tf

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector (or a factor of C) 
           of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, or if it does
    not have dim=3. Also, if it is not a multiple of the means
    passed in.
  """
  if image.get_shape().ndims % 3 != 0:
    raise ValueError('Input must be of size [height, width, C>0], C multiple of 3.')
  num_channels = image.get_shape().as_list()[-1]
  if num_channels % len(means) != 0:
    raise ValueError('len(means) must be a factor the number of channels.')
  means = means * int(num_channels / len(means))

  channels = tf.split(2, num_channels, image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(2, channels)
