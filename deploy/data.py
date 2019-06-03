from glob import glob

import numpy as np
from scipy.misc import imread

REGEX_FOR_INPUT_IMAGES = "/data/images-test/*.PNG"


def test_data():
    test_files = sorted(glob(REGEX_FOR_INPUT_IMAGES))

    X_test = []
    pids = []

    for f_path in test_files:
        pid = fname2pid(f_path)
        X_test.append(
            np.expand_dims(
                np.array(imread(f_path, flatten=False, mode="F")).astype(np.float32),
                axis=-1,
            )
        )
        pids.append(pid)

    X_test = np.array(X_test)

    X_test /= 255.
    X_test -= 0.5
    X_test *= 2.

    return X_test, pids


def fname2pid(fname):
    return fname.split("/")[-1].split(".")[0].lstrip("0")
