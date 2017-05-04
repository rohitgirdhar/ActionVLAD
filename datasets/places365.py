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

from datasets import dataset_utils
from datasets.image_read_utils import _decode_from_string
from datasets import dataset_utils

slim = tf.contrib.slim

_TRAIN_LIST = '/home/rgirdhar/Work/Data/020_Places365/places365_train_standard.txt'
_VAL_LIST = '/home/rgirdhar/Work/Data/020_Places365/places365_val.txt'

FLAGS = tf.app.flags.FLAGS

SPLITS_TO_SIZES = {'train': 1803460, 'val': 36500}

_NUM_CLASSES = 365

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [? x ? x 3] color image.',
    'label': 'A single integer between 0 and 364',
}


def readerFn(num_samples=1):
  class reader_func(tf.ReaderBase):
    @staticmethod
    def read(filename_queue):
      value = filename_queue.dequeue()
      fpath, label = tf.decode_csv(
          value, record_defaults=[[''], ['']],
          field_delim=' ')
      image_buffer = tf.read_file(fpath)
      return [image_buffer, label]
  return reader_func


def decoderFn(num_samples=1):
  class decoder_func(slim.data_decoder.DataDecoder):
    @staticmethod
    def list_items():
      return ['image', 'label']


    @staticmethod
    def decode(data, items):
      image_buffer = _decode_from_string(data)
      # if num_samples == 1:
        # tf.Assert(tf.shape(image_buffer)[0] == 1, image_buffer)
        # image_buffer = image_buffer[0]
      # else:
      image_buffer = tf.pack(image_buffer)
      return image_buffer
  return decoder_func


def get_split(split_name, dataset_dir, dataset_list_dir='', file_pattern=None, reader=None, modality='rgb', num_samples=1):
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  # if not file_pattern:
  #   file_pattern = _FILE_PATTERN
  # file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  if split_name == 'train':
    _LIST = _TRAIN_LIST
  else:
    _LIST = _VAL_LIST
  with open(_LIST, 'r') as fin:
    data_sources = [
      ' '.join([os.path.join(dataset_dir, el.split()[0]),] + el.split()[1:])
      for el in fin.read().splitlines()
    ]

  # Allowing None in the signature so that dataset_factory can use the default.
  if not reader:
    reader = readerFn

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=data_sources,
      reader=reader,
      decoder=decoderFn(num_samples),
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
