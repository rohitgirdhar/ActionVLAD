# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import tensorflow as tf

IM_HT = 256
IM_WD = 340

def _read_from_disk_spatial(fpath, nframes, num_samples=25, start_frame=0,
                            file_prefix='', file_zero_padding=4, file_index=1,
                            dataset_dir='', step=None):
    duration = nframes
    if step is None:
      if num_samples == 1:
          step = tf.random_uniform([1], 0, nframes, dtype='int32')[0]
      else:
          step = tf.cast((duration-tf.constant(1)) /
                         (tf.constant(num_samples-1)), 'int32')
    allimgs = []
    with tf.variable_scope('read_rgb_video'):
        for i in range(num_samples):
            if num_samples == 1:
                i = 1  # so that the random step value can be used
            with tf.variable_scope('read_rgb_image'):
                prefix = file_prefix + '_' if file_prefix else ''
                impath = tf.string_join([
                    tf.constant(dataset_dir + '/'),
                    fpath, tf.constant('/'),
                    prefix,
                    tf.as_string(start_frame + i * step + file_index,
                      width=file_zero_padding, fill='0'),
                    tf.constant('.jpg')])
                img_str = tf.read_file(impath)
            allimgs.append(img_str)
    return allimgs


def _read_from_disk_temporal(
    fpath, nframes, num_samples=25,
    optical_flow_frames=10, start_frame=0,
    file_prefix='', file_zero_padding=4, file_index=1,
    dataset_dir='', step=None):
    duration = nframes
    if step is None:
      if num_samples == 1:
          step = tf.random_uniform([1], 0, nframes-optical_flow_frames-1, dtype='int32')[0]
      else:
          step = tf.cast((duration-tf.constant(optical_flow_frames)) /
                         (tf.constant(num_samples)), 'int32')
    allimgs = []
    with tf.variable_scope('read_flow_video'):
        for i in range(num_samples):
            if num_samples == 1:
                i = 1  # so that the random step value can be used
            with tf.variable_scope('read_flow_image'):
              flow_img = []
              for j in range(optical_flow_frames):
                with tf.variable_scope('read_flow_channels'):
                  for dr in ['x', 'y']:
                    prefix = file_prefix + '_' if file_prefix else ''
                    impath = tf.string_join([
                        tf.constant(dataset_dir + '/'),
                        fpath, tf.constant('/'),
                        prefix, '%s_' % dr,
                        tf.as_string(start_frame + i * step + file_index + j,
                          width=file_zero_padding, fill='0'),
                        tf.constant('.jpg')])
                    img_str = tf.read_file(impath)
                    flow_img.append(img_str)
              allimgs.append(flow_img)
    return allimgs


def decode_rgb(img_str):
  with tf.variable_scope('decode_rgb_frame'):
    img = tf.image.decode_jpeg(img_str, channels=3)
    # Always convert before resize, this is a bug in TF
    # https://github.com/tensorflow/tensorflow/issues/1763
    # IMPORTANT NOTE: The original netvlad model was trained with the convert
    # happening after the resize, and hence it's trained with the large values.
    # It still works if I do that, but I'm training a new netvlad RGB model
    # with the current setup.
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, [IM_HT, IM_WD])
  return [img]


def decode_flow(img_str):
  # IMPORTANT NOTE: I am now resizing the flow frames before running through
  # the preprocessing. I was not doing that earlier (in the master). This leads
  # to the 66 number to drop to 63 on HMDB. But it should be fixable by
  # re-training with this setup
  with tf.variable_scope('decode_flow_frame'):
    img = tf.concat(2, [tf.image.decode_jpeg(el, channels=1)
      for el in tf.unpack(img_str)])
    # Always convert before resize, this is a bug in TF
    # https://github.com/tensorflow/tensorflow/issues/1763
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, [IM_HT, IM_WD])
  return [img]


def _decode_from_string(img_str, modality):
  if modality == 'rgb':
    return decode_rgb(img_str)
  elif modality.startswith('flow'):
    return decode_flow(img_str)
  elif modality.startswith('rgb+flow'):
    with tf.name_scope('decode_rgbNflow'):
      img_rgb = decode_rgb(img_str[..., 0])
      img_flow = decode_flow(img_str[..., 1:])
      return [img_rgb[0], img_flow[0]]
