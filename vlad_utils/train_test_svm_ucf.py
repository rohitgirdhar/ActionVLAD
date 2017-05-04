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
from sklearn.preprocessing import StandardScaler

# train_path = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/processed/002_Snapshots/002_NetTFSlimNetVLAD/028_VGG_UCF101_pretrained_netvlad/Features/train.h5'
train_path = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/processed/002_Snapshots/002_NetTFSlimNetVLAD/030_Debug/Features/train.h5'
train_list = '/home/rgirdhar/Work/Data/018_VideoVLAD/raw/UCF101/Lists/trainlist01_withCounts.txt'
# test_path = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/processed/002_Snapshots/002_NetTFSlimNetVLAD/028_VGG_UCF101_pretrained_netvlad/Features/test.h5'
test_path = '/nfs.yoda/rgirdhar/Work/Data2/018_VideoVLAD/processed/002_Snapshots/002_NetTFSlimNetVLAD/030_Debug/Features/test.h5'
test_list = '/home/rgirdhar/Work/Data/018_VideoVLAD/raw/UCF101/Lists/testlist01_withCounts.txt'

def readData(h5_path, txt_path, feature='netvlad'):
  with h5py.File(h5_path, 'r') as fin:
    feats = fin[feature].value
  with open(txt_path, 'r') as fin:
    labels = [int(el.split()[-1]) for el in fin.read().splitlines()]
  return feats, labels

train_feats, train_labels = readData(train_path, train_list)
test_feats, test_labels = readData(test_path, test_list)

scaler = StandardScaler()
train_feats = scaler.fit_transform(train_feats)
test_feats = scaler.transform(test_feats)

clf = sklearn.svm.LinearSVC(C=1)
# clf = learn.LinearClassifier(n_classes=101)
# clf = sklearn.svm.SVC(C=1, kernel='linear', probability=False, decision_function_shape='ovr')
# clf = sklearn.linear_model.LogisticRegression()
clf.fit(train_feats, train_labels)
del train_feats
# clf.fit(train_feats, train_labels, steps=20000, batch_size=32)

res = clf.predict(test_feats)

print 'acc: %f' % (np.mean(res == np.array(test_labels)))

