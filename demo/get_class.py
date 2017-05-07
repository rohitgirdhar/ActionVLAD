import h5py
import sys

# argv[1] : H5 file path
# argv[2] : path to actions list

with h5py.File(sys.argv[1], 'r') as fin, open(sys.argv[2], 'r') as fin2:
  act_names = fin2.read().splitlines()
  act = act_names[fin['stream0/logits'].value.argmax()]
  print('Detected action: {}'.format(act))
