import os
import sys
from glob import glob
from random import seed, randint

import numpy as np
import pandas as pd
import tensorflow as tf
from imgaug import augmenters
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from scipy.misc import imread
from sklearn.metrics import roc_auc_score

from focal_loss import focal_loss
from model import multitask_cnn

data_path = './data.csv'
images_dir = '/media/maciej/Thyroid/thyroid-nodules/images-cv'
checkpoints_dir = '/media/maciej/Thyroid/thyroid-nodules/multitask/custom/checkpoints/<FOLD>/'
logs_dir = '/media/maciej/Thyroid/thyroid-nodules/multitask/custom/logs/<FOLD>/'

random_seed = 3
total_folds = 10
batch_size = 128
epochs = 250
base_lr = 0.001


def validation_ids(fold, total_folds, df_cancer):
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


def train(fold):
	fold_checkpoints_dir = checkpoints_dir.replace('<FOLD>', str(fold))
	fold_logs_dir = logs_dir.replace('<FOLD>', str(fold))
	
	if not os.path.exists(fold_checkpoints_dir):
		os.makedirs(fold_checkpoints_dir)
	if not os.path.exists(fold_logs_dir):
		os.makedirs(fold_logs_dir)
	
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
	val_ids = validation_ids(fold, total_folds, df_cancer)
	
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
		if pid in val_ids:
			X_test.append(np.expand_dims(np.array(imread(f_path, flatten=False, mode='F')).astype(np.float32), axis=-1))
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
			X_train.append(np.expand_dims(np.array(imread(f_path, flatten=False, mode='F')).astype(np.float32), axis=-1))
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
	
	print(X_train.shape)
	print(X_test.shape)
	
	X_train /= 255.
	X_train -= 0.5
	X_train *= 2.
	
	X_test /= 255.
	X_test -= 0.5
	X_test *= 2.
	
	print('Training and validation data processed.')
	
	model = multitask_cnn()
	
	optimizer = RMSprop(lr=base_lr)
	model.compile(optimizer=optimizer,
	              loss={'out_cancer': focal_loss(),
	                    'out_compos': focal_loss(),
	                    'out_echo': focal_loss(),
	                    'out_shape': focal_loss(),
	                    'out_calcs': focal_loss(),
	                    'out_margin': focal_loss()},
	              loss_weights={'out_cancer': 1.0,
	                            'out_compos': 1.0,
	                            'out_echo': 1.0,
	                            'out_shape': 1.0,
	                            'out_calcs': 1.0,
	                            'out_margin': 1.0},
	              metrics=['accuracy'])
	
	training_log = TensorBoard(log_dir=os.path.join(fold_logs_dir, 'log'), write_graph=False)
	
	callbacks = [
		training_log,
	]
	
	for e in range(epochs):
		X_train_augmented = augment(X_train)
		model.fit({'thyroid_input': X_train_augmented},
		          {'out_cancer': np.array(y_train_cancer),
		           'out_compos': np.array(y_train_compos),
		           'out_echo': np.array(y_train_echo),
		           'out_shape': np.array(y_train_shape),
		           'out_calcs': np.array(y_train_calcs),
		           'out_margin': np.array(y_train_margin)},
		          validation_data=(X_test,
		                           [np.array(y_test_cancer), np.array(y_test_compos), np.array(y_test_echo),
		                            np.array(y_test_shape), np.array(y_test_calcs), np.array(y_test_margin)]),
		          batch_size=batch_size,
		          epochs=e + 1,
		          initial_epoch=e,
		          shuffle=True,
		          callbacks=callbacks)
		
		if np.mod(e + 1, 20) == 0:
			y_pred = model.predict(X_train, batch_size=batch_size, verbose=1)
			auc_train = roc_auc_score(np.array(y_train_cancer), y_pred[0])
			y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
			auc_test = roc_auc_score(np.array(y_test_cancer), y_pred[0])
			with open(os.path.join(fold_logs_dir, 'auc.txt'), 'a') as auc_file:
				auc_file.write('{},{}\n'.format(auc_train, auc_test))
		
		if np.mod(e + 1, 50) == 0:
			model.save(os.path.join(fold_checkpoints_dir, 'weights_{}.h5'.format(e + 1)))
	
	print('Training fold {} completed.'.format(fold))


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


def fname2pid(fname):
	return fname.split('/')[-1].split('.')[0].lstrip('0')


if __name__ == '__main__':
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	sess = tf.Session(config=config)
	K.set_session(sess)
	
	device = '/gpu:' + sys.argv[1]
	with tf.device(device):
		train(int(sys.argv[2]))
