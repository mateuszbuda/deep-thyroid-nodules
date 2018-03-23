import csv
import numpy as np
import os
from glob import glob

with open('train_ids.csv', 'rU') as f:
	reader = csv.reader(f)
	next(reader)
	train_ids = list(reader)

train_ids = set(np.squeeze(train_ids))

filenames = sorted(glob('/media/maciej/Thyroid/thyroid-nodules/Nodules/*.PNG'))

fids = set()

for fname in filenames:
	fid = fname.split('/')[-1].split('.')[0]
	fid = fid.lstrip('0')
	fids.add(fid)

diff = fids.difference(train_ids)

for fid in diff:
	parts = fid.split('_')
	parts[0] = parts[0].zfill(4)
	to_remove = glob('/media/maciej/Thyroid/thyroid-nodules/Nodules/{}*.PNG'.format('_'.join(parts)))
	for fname in to_remove:
		print('Removing {}'.format(fname))
		os.remove(fname)
