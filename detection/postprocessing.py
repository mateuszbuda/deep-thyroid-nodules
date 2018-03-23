import csv
import os
import sys
from glob import glob

import numpy as np

if len(sys.argv) == 1:
	path_regex = '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers-cv/*.csv'
else:
	path_regex = sys.argv[1]

filenames = glob(path_regex)

fid2fnames = {}

for fname in filenames:
	fid = fname.split('/')[-1].split('.')[0]
	fid2fnames.setdefault(fid, []).append(fname)

for fid in fid2fnames:
	assert len(fid2fnames[fid]) == 2
	
	cals0 = np.atleast_2d(np.genfromtxt(fid2fnames[fid][0], delimiter=','))
	cals1 = np.atleast_2d(np.genfromtxt(fid2fnames[fid][1], delimiter=','))
	
	cals0_len = len(cals0)
	cals1_len = len(cals1)
	
	sorted_len = sorted((cals0_len, cals1_len))
	
	if sorted_len == [2, 3]:
		continue
	
	if not (sorted_len == [2, 4] or sorted_len == [4, 4] or sorted_len == [2, 2]):
		print('Fixing {}'.format(fid))
		
		if cals0_len > 4:
			cals0 = cals0[0:4, :]
		if cals1_len > 4:
			cals1 = cals1[0:4, :]
		if len(cals0) == 4 and len(cals1) == 3:
			cals1 = cals1[0:2]
		if len(cals1) == 4 and len(cals0) == 3:
			cals0 = cals0[0:2]
		
		os.remove(fid2fnames[fid][0])
		with open(fid2fnames[fid][0], 'a') as f:
			writer = csv.writer(f)
			for row in cals0:
				writer.writerow(row)
		
		os.remove(fid2fnames[fid][1])
		with open(fid2fnames[fid][1], 'a') as f:
			writer = csv.writer(f)
			for row in cals1:
				writer.writerow(row)

print('\nDone')
