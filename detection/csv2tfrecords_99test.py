import hashlib
import io
import os
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
from glob import glob

from object_detection.utils import dataset_util

from random import randint, seed

flags = tf.app.flags
flags.DEFINE_string('train_record',
                    '/media/maciej/Thyroid/thyroid-nodules/detection/test/data/train.record',
                    'Path to output train TFRecord')
flags.DEFINE_string('test_record',
                    '/media/maciej/Thyroid/thyroid-nodules/detection/test/data/valid.record',
                    'Path to output validation TFRecord')
flags.DEFINE_string('csv_dir',
                    '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers',
                    'Path to input csv files')
flags.DEFINE_string('png_dir',
                    '/media/maciej/Thyroid/thyroid-nodules/detection/Nodules',
                    'Path to input PNG files')
flags.DEFINE_string('test_png_dir',
                    '/media/maciej/Thyroid/thyroid-nodules/detection/Nodules-test',
                    'Path to input PNG files')

FLAGS = flags.FLAGS


def create_tf_example(example):
	# example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}
	imagepath = example['imagepath']
	filename = example['filename']  # Filename of the image. Empty if image is not from file
	
	with tf.gfile.GFile(imagepath, 'rb') as fid:
		encoded_png = fid.read()
	encoded_png_io = io.BytesIO(encoded_png)
	im = Image.open(encoded_png_io)
	if im.format != 'PNG':
		raise ValueError('Image format not PNG')
	
	key = hashlib.sha256(encoded_png).hexdigest()
	
	width, height = im.size
	
	xmins = np.array(example['xmins']) / width
	xmins = np.maximum(xmins, 0.0)
	xmins = xmins.tolist()  # List of normalized left x coordinates in bounding box (1 per box)
	
	xmaxs = np.array(example['xmaxs']) / width
	xmaxs = np.minimum(xmaxs, 1.0)
	xmaxs = xmaxs.tolist()  # List of normalized right x coordinates in bounding box (1 per box)
	
	ymins = np.array(example['ymins']) / height
	ymins = np.maximum(ymins, 0.0)
	ymins = ymins.tolist()  # List of normalized top y coordinates in bounding box (1 per box)
	
	ymaxs = np.array(example['ymaxs']) / height
	ymaxs = np.minimum(ymaxs, 1.0)
	ymaxs = ymaxs.tolist()  # List of normalized bottom y coordinates in bounding box (1 per box)
	
	num_boxes = len(example['xmins'])
	label = example['label']
	# List of string class name of bounding box (1 per box)
	classes_text = ['cal'] * num_boxes
	classes = [label] * num_boxes  # List of integer class id of bounding box (1 per box)
	
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
		'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
		'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
		'image/encoded': dataset_util.bytes_feature(encoded_png),
		'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	
	return tf_example


def test_tfrecord():
	test_files = glob(os.path.join(FLAGS.test_png_dir, '*.PNG'))
	
	examples = {}
	# examples: {'imagepath':dict}
	for f_path in test_files:
		key = f_path
		filename = key.split('/')[-1]
		
		example = examples.get(key, {'imagepath': key, 'filename': filename})
		# example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}
		
		example['label'] = 1  # there's only one class: 1 - cal
		
		# dummy data - we don't have gt bboxes for test cases
		example['xmins'] = [10]
		example['ymins'] = [10]
		example['xmaxs'] = [20]
		example['ymaxs'] = [20]
		
		examples[key] = example
	
	test_writer = tf.python_io.TFRecordWriter(FLAGS.test_record)
	
	for key in examples:
		example = examples[key]
		# example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}
		tf_example = create_tf_example(example)
		test_writer.write(tf_example.SerializeToString())
	
	test_writer.close()


def train_tfrecord():
	csv_files = glob(os.path.join(FLAGS.csv_dir, '*.csv'))
	
	examples = {}
	# examples: {'imagepath':dict}
	for csv_f in csv_files:
		key = csv_f.replace(FLAGS.csv_dir, FLAGS.png_dir).replace('.csv', '.PNG')  # imagepath
		filename = key.split('/')[-1]
		
		example = examples.get(key, {'imagepath': key, 'filename': filename})
		# example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}
		
		example['label'] = 1  # there's only one class: 1 - cal
		
		xmins = example.get('xmins', [])
		ymins = example.get('ymins', [])
		xmaxs = example.get('xmaxs', [])
		ymaxs = example.get('ymaxs', [])
		
		for line in open(csv_f):
			values = line.rstrip().split(',')
			x = float(values[1])
			y = float(values[0])
			xmins.append(x - 8)
			ymins.append(y - 8)
			xmaxs.append(x + 8)
			ymaxs.append(y + 8)
		
		example['xmins'] = xmins
		example['ymins'] = ymins
		example['xmaxs'] = xmaxs
		example['ymaxs'] = ymaxs
		
		examples[key] = example
	
	train_writer = tf.python_io.TFRecordWriter(FLAGS.train_record)
	
	for key in examples:
		example = examples[key]
		# example: {'imagepath':string, 'filename':string, 'label':int, 'xmins':list, 'ymins':list, 'xmaxs':list, 'ymaxs':list}}
		tf_example = create_tf_example(example)
		train_writer.write(tf_example.SerializeToString())
	
	train_writer.close()
	

def main(_):
	train_tfrecord()
	test_tfrecord()


if __name__ == '__main__':
	tf.app.run()
