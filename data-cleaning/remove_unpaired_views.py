import os
from glob import glob


filenames = sorted(glob('/media/maciej/Thyroid/thyroid-nodules/Nodules/*.PNG'))

fid2fnames = {}

for fname in filenames:
	fid = fname.split('/')[-1].split('.')[0]
	fid2fnames.setdefault(fid, []).append(fname)

for fid in fid2fnames:
	if not len(fid2fnames[fid]) == 2:
		for fname in fid2fnames[fid]:
			print('Removing {}'.format(fname))
			os.remove(fname)
