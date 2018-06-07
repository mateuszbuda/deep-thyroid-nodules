import csv
import os
from glob import glob

import numpy as np

test_dir = "/media/maciej/Thyroid/thyroid-nodules/Nodules-test/"

with open("test_ids.csv", "rU") as f:
    reader = csv.reader(f)
    next(reader)
    test_ids = list(reader)

test_ids = np.squeeze(test_ids)

filenames = sorted(glob("/media/maciej/Thyroid/thyroid-nodules/Nodules/*.PNG"))

fid2fnames = {}

for fname in filenames:
    fid = fname.split("/")[-1].split(".")[0]
    fid2fnames.setdefault(fid, []).append(fname)

for fid in fid2fnames:
    if fid in test_ids:
        for fname in fid2fnames[fid]:
            dst_path = os.path.join(test_dir, os.path.split(fname)[1])
            os.rename(fname, dst_path)
