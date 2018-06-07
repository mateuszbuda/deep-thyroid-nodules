import hashlib
import io
import os
import sys
from glob import glob
from random import randint, seed

import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

fold = int(sys.argv[1])

flags = tf.app.flags
flags.DEFINE_string(
    "train_record",
    "/media/maciej/Thyroid/thyroid-nodules/detection/{}/data/train.record".format(fold),
    "Path to output train TFRecord",
)
flags.DEFINE_string(
    "test_record",
    "/media/maciej/Thyroid/thyroid-nodules/detection/{}/data/valid.record".format(fold),
    "Path to output validation TFRecord",
)
flags.DEFINE_string(
    "csv_dir",
    "/media/maciej/Thyroid/thyroid-nodules/detection/Calipers",
    "Path to input csv files",
)
flags.DEFINE_string(
    "png_dir",
    "/media/maciej/Thyroid/thyroid-nodules/detection/Nodules",
    "Path to input PNG files",
)

FLAGS = flags.FLAGS


def create_tf_example(example):
    # example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}
    imagepath = example["imagepath"]
    # Filename of the image. Empty if image is not from file
    filename = example["filename"]

    with tf.gfile.GFile(imagepath, "rb") as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    im = Image.open(encoded_png_io)
    if im.format != "PNG":
        raise ValueError("Image format not PNG")

    key = hashlib.sha256(encoded_png).hexdigest()

    width, height = im.size

    xmins = np.array(example["xmins"]) / width
    xmins = np.maximum(xmins, 0.0)
    # List of normalized left x coordinates in bounding box (1 per box)
    xmins = xmins.tolist()

    xmaxs = np.array(example["xmaxs"]) / width
    xmaxs = np.minimum(xmaxs, 1.0)
    # List of normalized right x coordinates in bounding box (1 per box)
    xmaxs = xmaxs.tolist()

    ymins = np.array(example["ymins"]) / height
    ymins = np.maximum(ymins, 0.0)
    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = ymins.tolist()

    ymaxs = np.array(example["ymaxs"]) / height
    ymaxs = np.minimum(ymaxs, 1.0)
    # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs = ymaxs.tolist()

    num_boxes = len(example["xmins"])
    label = example["label"]
    # List of string class name of bounding box (1 per box)
    classes_text = ["cal"] * num_boxes
    # List of integer class id of bounding box (1 per box)
    classes = [label] * num_boxes

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename.encode("utf8")),
                "image/source_id": dataset_util.bytes_feature(filename.encode("utf8")),
                "image/key/sha256": dataset_util.bytes_feature(key.encode("utf8")),
                "image/encoded": dataset_util.bytes_feature(encoded_png),
                "image/format": dataset_util.bytes_feature("png".encode("utf8")),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )

    return tf_example


def main(_):
    csv_files = glob(os.path.join(FLAGS.csv_dir, "*.csv"))

    # set of unique patient ids used for patient based train/validation split
    pid_set = set()

    examples = {}
    # examples: {'imagepath':dict}
    for csv_f in csv_files:
        # imagepath
        key = csv_f.replace(FLAGS.csv_dir, FLAGS.png_dir).replace(".csv", ".PNG")
        filename = key.split("/")[-1]

        example = examples.get(key, {"imagepath": key, "filename": filename})
        # example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}

        pid = fname2pid(filename)
        pid_set.add(pid)
        example["label"] = 1  # there's only one class: 1 - cal

        xmins = example.get("xmins", [])
        ymins = example.get("ymins", [])
        xmaxs = example.get("xmaxs", [])
        ymaxs = example.get("ymaxs", [])

        for line in open(csv_f):
            values = line.rstrip().split(",")
            x = float(values[1])
            y = float(values[0])
            xmins.append(x - 8)
            ymins.append(y - 8)
            xmaxs.append(x + 8)
            ymaxs.append(y + 8)

        example["xmins"] = xmins
        example["ymins"] = ymins
        example["xmaxs"] = xmaxs
        example["ymaxs"] = ymaxs

        examples[key] = example

    seed(42)
    test_folds = {}  # maps pid to its test fold number in 10-fold cross-validation split
    for pid in sorted(pid_set):
        test_folds[pid] = randint(0, 9)

    train_writer = tf.python_io.TFRecordWriter(FLAGS.train_record)
    test_writer = tf.python_io.TFRecordWriter(FLAGS.test_record)

    for key in examples:
        example = examples[key]
        # example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}
        tf_example = create_tf_example(example)
        pid = fname2pid(example["filename"])
        if test_folds[pid] == fold:
            test_writer.write(tf_example.SerializeToString())
        else:
            train_writer.write(tf_example.SerializeToString())

    train_writer.close()
    test_writer.close()


def fname2pid(fname):
    return fname.split("/")[-1].split(".")[0].lstrip("0")


if __name__ == "__main__":
    tf.app.run()
