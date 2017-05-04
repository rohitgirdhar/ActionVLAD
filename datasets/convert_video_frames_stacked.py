# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import os
import cv2
import math
import numpy as np

if 1:
  vidlist = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/raw/HMDB51/lists/train_test_lists2/All.txt'
  if 1:
    modality = 'flow'
    imgdir = '/scratch/rgirdhar/Work/Data/018_VideoVLAD/raw/HMDB51/flow2'
    outdir = '/scratch/rgirdhar/Work/Data/018_VideoVLAD/raw/HMDB51/flow3_stacked'
elif 0:
  vidlist = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/raw/HMDB51/lists/train_test_lists2/All.txt'
  if 1:
    modality = 'rgb'
    imgdir = '/scratch/rgirdhar/Work/Data/018_VideoVLAD/raw/HMDB51/frames2'
    outdir = '/scratch/rgirdhar/Work/Data/018_VideoVLAD/raw/HMDB51/frames3_stacked'


def get_img_rgb(vpath, nframes, num_samples=25):
  duration = nframes
  step = int(math.floor((duration-1)/(num_samples-1)))
  imgs = []
  for i in range(num_samples):
    impath = os.path.join(imgdir, vpath, 'image_%05d.jpg' % (i*step+1))
    imgs.append(cv2.resize(cv2.imread(impath), (340, 256)))
  return [np.vstack(imgs)]


def get_img_flow(vpath, nframes, num_samples=25, optical_flow_frames=10):
  duration = nframes
  step = int(math.floor((duration-optical_flow_frames)/(num_samples)))
  imgs = []
  for i in range(num_samples):
    subimg = []
    for j in range(optical_flow_frames):
      for d in ['x', 'y']:
        impath = os.path.join(imgdir, vpath, 'flow_%c_%05d.jpg' % (d, i*step+j+1))
        I = cv2.imread(impath, 0)
        if I is None:
          print 'couldnt read %s' % impath
          I = np.ones((340, 256)) * 128
        subimg.append(cv2.resize(I, (340, 256)))
    imgs.append(np.vstack(subimg))
  return imgs


with open(vidlist, 'r') as fin:
  for lno,line in enumerate(fin):
    vpath, nframes, _ = line.split()
    nframes = int(nframes)
    outfpath = os.path.join(outdir, vpath, 'image.jpg')
    # if not locker.lock(outfpath):
    #   continue
    if modality == 'flow':
      img = get_img_flow(vpath, nframes, 25, optical_flow_frames=10)
    else:
      img = get_img_rgb(vpath, nframes, 25)
    try:
      os.makedirs(os.path.join(outdir, vpath))
    except:
      pass
    if len(img) == 1:
      cv2.imwrite(outfpath, img[0])
    else:
      for imid,im in enumerate(img):
        cv2.imwrite(os.path.join(outdir, vpath, 'image%d.jpg' % imid), im)
    print('Done %d' % (lno))
    # locker.unlock(outfpath)
