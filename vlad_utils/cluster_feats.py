# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import h5py
import argparse


def main():
    parser = argparse.ArgumentParser(description='Cluster Features')
    parser.add_argument('-k', '--nclusters',
            type=int, default=32,
            help='Number of clusters.')
    parser.add_argument('-j', '--njobs',
            type=int, default=8,
            help='Number of jobs to run.')
    parser.add_argument('-o', '--outfpath',
            type=str, required=True,
            help='Path to pkl file to store the clusters.')
    parser.add_argument('-i', '--inputfeatpath',
            type=str, required=True,
            help='Path to h5 file with features.')
    parser.add_argument('-n', '--feat_name',
            type=str, required=True,
            help='Layer name whose features to use.') 
    parser.add_argument('--nfeats', type=int,
            default=-1, help='Set to use subset of feats for clustering.')
    parser.add_argument('--relu', type=bool,
            default=False, help='Set true to relu the features after reading.')
    args = vars(parser.parse_args())

    with h5py.File(args['inputfeatpath'], 'r') as fin:
      allfeats = fin[args['feat_name']].value
    allfeats = allfeats[:args['nfeats'], ...]
    if args['relu']:
      print('ReLU-ing all features')
      allfeats[allfeats < 0] = 0
    allfeats = np.reshape(allfeats, (-1, allfeats.shape[-1]))
    print('Clustering %d feats of %d dim into %d clusters' % (
      allfeats.shape[0], allfeats.shape[1], args['nclusters']))
    # V.IMP to normalize (since the network sees normalized conv output)
    # Previous resnet code was not doing it, which might be a bug
    allfeats = normalize(allfeats)
    kmeans = KMeans(args['nclusters'], n_jobs=args['njobs'])
    kmeans.fit(allfeats)
    with open(args['outfpath'], 'wb') as fout:
        pickle.dump(kmeans, fout)


if __name__ == '__main__':
    main()
