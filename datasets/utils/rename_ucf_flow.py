# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
## Utility to convert downloaded Limin Wang flow into the same format as I use
## with rgb images
## http://mmlab.siat.ac.cn/very_deep_two_stream_model/ucf101_flow_img_tvl1_gpu.zip

import subprocess
import shutil
import os

lwang_flow_dir = \
  '/scratch/rgirdhar/Datasets/Video/001_UCF101/processed/Flow/ucf101_flow_img_tvl1_gpu/'
out_flow_dir = \
  '/scratch/rgirdhar/Datasets/Video/001_UCF101/processed/Flow/renamed/'

vidlist = \
  '/data/rgirdhar/Data2/Projects/2016/001_NetVLADVideo/raw/UCF101/Lists/AllVideos_withLabel.txt'

buggy = ['Rafting/v_Rafting_g01_c02.avi']

with open(vidlist, 'r') as fin:
  for line in fin:
    fname, count, _ = line.split()
    count = int(count)
    if fname in buggy:
      continue
    outdir = '%s/%s' % (out_flow_dir, fname)
    if os.path.exists(outdir):
      continue
    subprocess.call('mkdir -p %s' % (outdir), shell=True)
    for i in range(1, count):
      for d in ['x', 'y']:
        shutil.copyfile('%s/%s/flow_%c_%04d.jpg' % (
          lwang_flow_dir, fname[:-4], d, i),
          '%s/%s/flow_%c_%05d.jpg' % (
            out_flow_dir, fname, d, i))

# and rename the buggy one in bash
# for i in {0..211}; do cur=`printf "%04d_y.jpg" $i`; target=`printf "flow_y_%05d.jpg" $(($i+1))`; mv $cur $target; done
