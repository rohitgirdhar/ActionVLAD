# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
### This script is only existing for legacy purposes. Use combine_streams2.py
### future experiments


import numpy as np
import h5py
import argparse


def main():
    parser = argparse.ArgumentParser(description='Cluster Features')
    parser.add_argument('-s', '--spatial',
            type=str, required=True,
            help='Path to h5 file with spatial fc8.')
    parser.add_argument('-t', '--temporal',
            type=str, required=True,
            help='Path to h5 file with temporal fc8.')
    parser.add_argument('--feat_name_temporal',
            type=str, default='stream0/logits',
            help='Layer name whose features to use from temporal.')
    parser.add_argument('--feat_name_spatial',
            type=str, default='stream0/logits',
            help='Layer name whose features to use from spatial.')
    parser.add_argument('-f', '--label_file',
            type=str, required=True,
            help='Test file with labels (UCF format).')
    parser.add_argument('-r', '--temporal_ratio',
            type=float, default=0.667,
            help='Weight for the temporal stream output.')
    parser.add_argument('--idt_scores',
            type=str, default=None,
            help='H5 file with IDT scores, from Gul.')
    parser.add_argument('--idt_wt',
            type=float, default=0.25,
            help='Wt on the L2 normalized IDT scores, compared to my score.')

    args = vars(parser.parse_args())

    with open(args['label_file'], 'r') as fin:
      labels = np.array([int(el.split()[-1]) for el in
                         fin.read().splitlines()])

    with h5py.File(args['spatial'], 'r') as fin:
      spatial = fin[args['feat_name_spatial']].value
    with h5py.File(args['temporal'], 'r') as fin:
      temporal = fin[args['feat_name_temporal']].value
    final = (spatial * (1.0 - args['temporal_ratio']) \
             + temporal * args['temporal_ratio'])
    acc_spat = np.mean(spatial.argmax(axis=1) == labels)
    acc_temp = np.mean(temporal.argmax(axis=1) == labels)
    acc = np.mean(final.argmax(axis=1) == labels)
    if args['idt_scores'] is not None:
      with h5py.File(args['idt_scores'], 'r') as fin:
        idt = fin['idt'].value
    else:
      idt = np.zeros(final.shape)
    acc_onlyIDT = np.mean(idt.argmax(axis=1) == labels)

    idt_wt = args['idt_wt']
    final = (1-idt_wt) * (
      final / np.linalg.norm(final, axis=1, keepdims=True)) + idt_wt * (
        idt / np.linalg.norm(idt, axis=1, keepdims=True))
    acc_withIDT = np.mean(final.argmax(axis=1) == labels)
    print('Spatial = %0.6f [*%f]\nTemporal = %0.6f [*%f]\nFinal acc = '
          '%0.6f.\nonly IDT = %f\nwith IDT = %f' %
          (acc_spat, (1.0 - args['temporal_ratio']), acc_temp, args['temporal_ratio'],
           acc, acc_onlyIDT, acc_withIDT))


if __name__ == '__main__':
    main()
