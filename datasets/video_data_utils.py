# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
"""Provides data for the UCF101 dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import sys

from datasets import dataset_utils
from datasets.image_read_utils import _read_from_disk_spatial, \
       _decode_from_string, _read_from_disk_temporal
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim


def getReaderFn(num_samples, modality='rgb', dataset_dir=''):
  def readerFn():
    class reader_func(tf.ReaderBase):
      @staticmethod
      def read(filename_queue):
        value = filename_queue.dequeue()
        fpath, nframes, label = tf.decode_csv(
            value, record_defaults=[[''], [-1], ['']],
            field_delim=' ')
        # TODO(rgirdhar): Release the file_prefix='', file_zero_padding=4,
        # file_index=1 options to the bash script
        if modality == 'rgb':
          assert(len(dataset_dir) >= 1)
          image_buffer = _read_from_disk_spatial(
              fpath, nframes, num_samples=num_samples,
              file_prefix='image', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[0])
        elif modality.startswith('flow'):
          assert(len(dataset_dir) >= 1)
          optical_flow_frames = int(modality[4:])
          image_buffer = _read_from_disk_temporal(
              fpath, nframes, num_samples=num_samples,
              optical_flow_frames=optical_flow_frames,
              file_prefix='flow', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[0])
        elif modality.startswith('rgb+flow'):
          assert(len(dataset_dir) >= 2)
          # in this case, fix the step for both the streams to ensure correspondence
          optical_flow_frames = int(modality[-2:])
          duration = nframes
          step = None
          if num_samples == 1:
            step = tf.random_uniform([1], 0, nframes-optical_flow_frames-1, dtype='int32')[0]
          else:
            step = tf.cast((duration-tf.constant(optical_flow_frames)) /
                           (tf.constant(num_samples)), 'int32')

          rgb_image_buffer = _read_from_disk_spatial(
              fpath, nframes, num_samples=num_samples,
              file_prefix='image', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[0],
              step=step)
          flow_image_buffer = _read_from_disk_temporal(
              fpath, nframes, num_samples=num_samples,
              optical_flow_frames=optical_flow_frames,
              file_prefix='flow', file_zero_padding=5, file_index=1,
              dataset_dir=dataset_dir[1],
              step=step)
          image_buffer = zip(rgb_image_buffer, flow_image_buffer)
          image_buffer = [[el[0]] + el[1] for el in image_buffer]
        else:
          logging.error('Unknown modality %s\n' % modality)
          raise ValueError()
        return [tf.pack(image_buffer), label]
    return reader_func
  return readerFn


def decoderFn(num_samples=1, modality='rgb'):
  class decoder_func(slim.data_decoder.DataDecoder):
    @staticmethod
    def list_items():
      return ['image', 'label']

    @staticmethod
    def decode(data, items):
      with tf.name_scope('decode_video'):
        if modality == 'rgb':
          data.set_shape((num_samples,))
        elif modality.startswith('flow'):
          optical_flow_frames = int(modality[4:])
          data.set_shape((num_samples, 2 * optical_flow_frames))
        elif modality.startswith('rgb+flow'):
          optical_flow_frames = int(modality[-2:])
          data.set_shape((num_samples, 1 + 2 * optical_flow_frames))
        else:
          logging.error('Unknown modality %s\n' % modality)
        image_buffer = [_decode_from_string(el, modality) for
                        el in tf.unpack(data)]
        # image_buffer = tf.pack(image_buffer)
        return image_buffer
  return decoder_func


def count_frames_file(fpath, frameLevel=True):
  res = 0
  with open(fpath, 'r') as fin:
    for line in fin:
      if frameLevel:
        res += int(line.split()[1])
      else:
        res += 1
  return res


def gen_dataset(split_name, dataset_dir, file_pattern=None,
                reader=None, modality='rgb', num_samples=1,
                split_id=1, num_classes=0, list_fn=None):
  SPLITS_TO_SIZES = {
    'train': count_frames_file(list_fn('train', split_id), frameLevel=(num_samples==1)),
    'test': count_frames_file(list_fn('test', split_id), frameLevel=(num_samples==1)),
  }
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  _ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [? x ? x 3] color image.',
    'label': 'A single integer between 0 and %d' % num_classes,
  }
  LIST_FILE = list_fn(split_name, split_id)
  logging.info('Using file %s' % LIST_FILE)
  with open(LIST_FILE, 'r') as fin:
    data_sources = fin.read().splitlines()

  # Allowing None in the signature so that dataset_factory can use the default.
  if not reader:
    reader = getReaderFn(num_samples, modality, dataset_dir)

  labels_to_names = None
  # if dataset_utils.has_labels(dataset_dir):
  #   labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=data_sources,
      reader=reader,
      decoder=decoderFn(num_samples, modality),
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=num_classes,
      labels_to_names=labels_to_names)
