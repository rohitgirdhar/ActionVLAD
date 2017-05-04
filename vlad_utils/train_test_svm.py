# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import sklearn
import sklearn.linear_model
import numpy as np
import h5py
import subprocess

train_path = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/processed/002_Snapshots/002_NetTFSlimNetVLAD/007_VGG_Places365_pretrained/Features/v2/netvlad_beforeReLU_k32.h5'
val_path = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/processed/002_Snapshots/002_NetTFSlimNetVLAD/007_VGG_Places365_pretrained/Features/v2/netvlad_beforeReLU_k32_VAL.h5'
MAX_TRAIN_SAMPLES = 30000

with h5py.File(train_path, 'r') as fin:
  train_feats = fin['feats'].value[:MAX_TRAIN_SAMPLES, ...]
  train_labels = fin['labels'].value[:MAX_TRAIN_SAMPLES, ...]

clf = sklearn.svm.LinearSVC(C=1)
# clf = learn.LinearClassifier(n_classes=101)
# clf = sklearn.svm.SVC(C=1, kernel='linear', probability=False, decision_function_shape='ovr')
# clf = sklearn.linear_model.LogisticRegression()
clf.fit(train_feats, train_labels)
del train_feats
# clf.fit(train_feats, train_labels, steps=20000, batch_size=32)

with h5py.File(val_path, 'r') as fin:
  val_feats = fin['feats'].value
  val_labels = fin['labels'].value
res = clf.predict(val_feats)

print 'acc: %f' % (np.mean(res == np.array(val_labels)))

subprocess.call('mkdir -p /tmp/clf_stor/', shell=True)
clf.save('/tmp/clf_stor')

