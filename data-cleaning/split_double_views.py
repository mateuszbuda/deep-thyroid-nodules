import os
import imageio
import numpy as np
from glob import glob


filenames = sorted(glob('/media/maciej/Thyroid/thyroid-nodules/Nodules/*.PNG'))

fid2fnames = {}

for fname in filenames:
	fid = fname.split('/')[-1].split('.')[0]
	fid2fnames.setdefault(fid, []).append(fname)

for fid in fid2fnames:
	assert len(fid2fnames[fid]) == 2
	
	image0 = imageio.imread(fid2fnames[fid][0])
	image1 = imageio.imread(fid2fnames[fid][1])
	
	if not image0.shape == image1.shape:
		continue

	if np.max(np.abs(image0 - image1)) > 1.0:
		continue

	# find split point
	image = np.mean(image0, axis=2)
	x_center = image.shape[1] / 2
	window = image.shape[1] / 10
	hor_proj = np.mean(image[:, x_center-window:x_center+window], axis=0)
	x_split = x_center - window + np.argmin(hor_proj)

	# get splitted images
	image_left = image0[:, :x_split, :]
	image_right = image0[:, x_split:, :]

	# remove original double-view images
	os.remove(fid2fnames[fid][0])
	os.remove(fid2fnames[fid][1])

	# save splitted images
	path_left = fid2fnames[fid][0].replace('trans', 'left').replace('long', 'left')
	path_right = path_left.replace('left', 'right')
	imageio.imwrite(path_left, image_left)
	imageio.imwrite(path_right, image_right)
