import os
import numpy as np
from glob import glob


filenames = sorted(glob('/media/maciej/Thyroid/thyroid-nodules/detection/Calipers/*.csv'))
images_dir = 'Nodules'

fid2fnames = {}

for fname in filenames:
	fid = fname.split('/')[-1].split('.')[0]
	fid2fnames.setdefault(fid, []).append(fname)

unpaired_cnt = 0

for fid in sorted(fid2fnames):
	assert len(fid2fnames[fid]) == 2
	
	cals0 = np.atleast_2d(np.genfromtxt(fid2fnames[fid][0], delimiter=','))
	cals1 = np.atleast_2d(np.genfromtxt(fid2fnames[fid][1], delimiter=','))
	
	cals0_len = len(cals0)
	cals1_len = len(cals1)
	
	if not (sorted((cals0_len, cals1_len)) == [2, 4] or
			        sorted((cals0_len, cals1_len)) == [4, 4] or
			        sorted((cals0_len, cals1_len)) == [2, 2]):
		unpaired_cnt += 1
		for fname in fid2fnames[fid]:
			print('Removing {}'.format(fname))
			os.remove(fname)
			image_fname = fname.replace('detection/Calipers', images_dir).replace('csv', 'PNG')
			print('Removing {}'.format(image_fname))
			os.remove(image_fname)
			image_fname = fname.replace('Calipers', images_dir).replace('csv', 'PNG')
			print('Removing {}'.format(image_fname))
			os.remove(image_fname)

print('\nRemoved {} cases with unpaired calipers'.format(unpaired_cnt))
