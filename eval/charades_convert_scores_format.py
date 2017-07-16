import h5py
import argparse
import numpy as np
import sklearn.preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--scores', help="H5 file with the scores")
parser.add_argument('--test', help="file with test file ids")
parser.add_argument('--idt', default=None, help="IDT scores file, shared by Gunnar")
parser.add_argument('--idt_wt', default=0.5, type=float, help="Weight to use for iDT.")
parser.add_argument('--outfpath', help="output file")
args = parser.parse_args()

with open(args.test, 'r') as fin:
  vid_ids = [line.split()[0][2:-4] for line in fin.read().splitlines()]
with h5py.File(args.scores) as fin:
  scores = fin['stream0/logits'].value
if args.idt is not None and len(args.idt) > 0:
  with open(args.idt, 'r') as fin:
    print('Combining with iDT scores')
    lines = fin.read().splitlines()
    idt_vid_ids = [line.split()[0] for line in lines]
    idt_scores = np.array([[float(el) for el in line.split()[1:]] for line in
                           lines])
    # add 0 scores for videos for which there is no iDT score
    missing_vids = [el for el in vid_ids if el not in idt_vid_ids]
    print('%d missing videos scores in iDT. Using 0s for those.' % len(missing_vids))
    idt_scores = np.vstack((idt_scores, np.zeros((len(missing_vids),
                                                  idt_scores.shape[1]))))
    idt_vid_ids = idt_vid_ids + missing_vids
    order = [idt_vid_ids.index(el) for el in vid_ids]
    idt_scores = idt_scores[np.array(order), :]
    scores = args.idt_wt * sklearn.preprocessing.normalize(idt_scores) + \
             (1 - args.idt_wt) * sklearn.preprocessing.normalize(scores)

with open(args.outfpath, 'w') as fout:
  for i in range(len(vid_ids)):
    fout.write('%s %s\n' % (vid_ids[i], ' '.join([str(el) for el in scores[i, :]])))
