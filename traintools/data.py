from glob import glob
from random import seed, randint

import numpy as np
import pandas as pd
from imgaug import augmenters
from scipy.misc import imread

data_path = './data.csv'
images_dir = '/media/maciej/Thyroid/thyroid-nodules/images-cv'

random_seed = 3
total_folds = 10
batch_size = 128
epochs = 250
base_lr = 0.001


def fold_pids(fold, test=True):
	df = pd.read_csv(data_path)
	all_files = glob(images_dir + '/*.PNG')
	val_ids = validation_ids(fold, df[['ID', 'Cancer']])
	
	pids = []
	
	for f_path in all_files:
		pid = fname2pid(f_path)
		if (test and pid in val_ids) or (not test and pid not in val_ids):
			pids.append(pid)
	
	return pids


def data(fold):
	df = pd.read_csv(data_path)
	df.fillna(0, inplace=True)
	df.Calcs1.replace(0, 'None', inplace=True)
	
	df_cancer = df[['ID', 'Cancer']]
	df_compos = pd.concat([df.ID, pd.get_dummies(df.Composition)], axis=1)
	df_echo = pd.concat([df.ID, pd.get_dummies(df.Echogenicity)], axis=1)
	df_shape = df[['ID', 'Shape']]
	df_shape['Shape'] = df_shape.apply(lambda row: 1 if row.Shape == 'y' else 0, axis=1)
	df_calcs = pd.concat([df.ID, pd.get_dummies(df.Calcs1)], axis=1)
	df_margin = pd.concat([df.ID, pd.get_dummies(df.MargA)], axis=1)
	
	all_files = glob(images_dir + '/*.PNG')
	val_ids = validation_ids(fold, df_cancer)
	
	X_train = []
	X_test = []
	
	y_train_cancer = []
	y_train_compos = []
	y_train_echo = []
	y_train_shape = []
	y_train_calcs = []
	y_train_margin = []
	y_test_cancer = []
	y_test_compos = []
	y_test_echo = []
	y_test_shape = []
	y_test_calcs = []
	y_test_margin = []
	
	for f_path in all_files:
		pid = fname2pid(f_path)
		image = np.expand_dims(np.array(imread(f_path, flatten=False, mode='F')).astype(np.float32), axis=-1)
		if pid in val_ids:
			X_test.append(image)
			y_test_cancer.append(df_cancer[df_cancer.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_compos.append(df_compos[df_compos.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_echo.append(df_echo[df_echo.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			if 'trans' in f_path:
				y_test_shape.append(df_shape[df_shape.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			else:
				y_test_shape.append(np.array([0]).astype(np.float32))
			y_test_calcs.append(df_calcs[df_calcs.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_margin.append(df_margin[df_margin.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
		else:
			X_train.append(image)
			y_train_cancer.append(df_cancer[df_cancer.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_train_compos.append(df_compos[df_compos.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_train_echo.append(df_echo[df_echo.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			if 'trans' in f_path:
				y_train_shape.append(df_shape[df_shape.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			else:
				y_train_shape.append(np.array([0]).astype(np.float32))
			y_train_calcs.append(df_calcs[df_calcs.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_train_margin.append(df_margin[df_margin.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
	
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	
	X_train /= 255.
	X_train -= 0.5
	X_train *= 2.
	
	X_test /= 255.
	X_test -= 0.5
	X_test *= 2.
	
	y_train = {
		'out_cancer': np.array(y_train_cancer),
		'out_compos': np.array(y_train_compos),
		'out_echo': np.array(y_train_echo),
		'out_shape': np.array(y_train_shape),
		'out_calcs': np.array(y_train_calcs),
		'out_margin': np.array(y_train_margin)
	}
	
	y_test = [
		np.array(y_test_cancer),
		np.array(y_test_compos),
		np.array(y_test_echo),
		np.array(y_test_shape),
		np.array(y_test_calcs),
		np.array(y_test_margin)
	]
	
	return X_train, y_train, X_test, y_test


def augment(X):
	seq = augmenters.Sequential([
		augmenters.Fliplr(0.5),
		augmenters.Flipud(0.5),
		augmenters.Affine(rotate=(-15, 15)),
		augmenters.Affine(shear=(-15, 15)),
		augmenters.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}),
		augmenters.Affine(scale=(0.9, 1.1))
	])
	return seq.augment_images(X)


def validation_ids(fold, df_cancer):
	pid_set = set()
	all_files = glob(images_dir + '/*.PNG')
	for f_path in all_files:
		pid = fname2pid(f_path)
		pid_set.add(pid)
	
	val_ids = []
	
	seed(random_seed)
	malignant_fold = 0
	for pid in sorted(pid_set):
		label = df_cancer[df_cancer.ID == pid].as_matrix().flatten()[1:]
		if label == 1:
			if fold == np.mod(malignant_fold, total_folds):
				val_ids.append(pid)
			malignant_fold += 1
		else:
			if fold == randint(0, total_folds - 1):
				val_ids.append(pid)
	
	return val_ids


def fname2pid(fname):
	return fname.split('/')[-1].split('.')[0].lstrip('0')
