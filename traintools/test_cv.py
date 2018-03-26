import csv
import os
import sys
from glob import glob
from random import seed, randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from scipy.misc import imread
from sklearn.metrics import roc_auc_score, roc_curve

from focal_loss import focal_loss

data_path = './data.csv'
images_dir = '/media/maciej/Thyroid/thyroid-nodules/images-cv'
random_seed = 3
epoch = 250
checkpoints_dir = '/media/maciej/Thyroid/thyroid-nodules/multitask/custom/checkpoints/<FOLD>/'
batch_size = 128
nb_categories = 1


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


def predict(fold, total_folds, test_cases=True):
	fold_checkpoints_dir = checkpoints_dir.replace('<FOLD>', str(fold))
	weights_path = os.path.join(fold_checkpoints_dir, 'weights_{}.h5'.format(epoch))
	
	net = load_model(weights_path, custom_objects={'focal_loss_fixed': focal_loss()})
	
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
	
	X_test = []
	
	y_test_cancer = []
	y_test_compos = []
	y_test_echo = []
	y_test_shape = []
	y_test_calcs = []
	y_test_margin = []
	
	test_files = []
	
	for f_path in all_files:
		pid = fname2pid(f_path)
		append = pid in val_ids if test_cases else pid not in val_ids
		if append:
			test_files.append(f_path)
			X_test.append(np.expand_dims(np.array(imread(f_path, flatten=False, mode='F')).astype(np.float32), axis=-1))
			y_test_cancer.append(df_cancer[df_cancer.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_compos.append(df_compos[df_compos.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_echo.append(df_echo[df_echo.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_shape.append(df_shape[df_shape.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_calcs.append(df_calcs[df_calcs.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
			y_test_margin.append(df_margin[df_margin.ID == pid].as_matrix().flatten()[1:].astype(np.float32))
	
	X_test = np.array(X_test)
	
	print(X_test.shape)
	
	X_test /= 255.
	X_test -= 0.5
	X_test *= 2.
	
	preds = net.predict(X_test, batch_size=batch_size, verbose=1)
	
	return test_files, preds[0], y_test_cancer


def test(folds):
	filenames = []
	predictions = np.zeros((0, nb_categories))
	targets = []
	pid_fold = []
	
	for f in range(folds):
		fnames, preds, t = predict(f, folds)
		predictions = np.vstack((predictions, preds))
		filenames.extend(fnames)
		targets.extend(t)
		pid_fold.extend([f] * len(targets))
	
	print('{} images'.format(len(filenames)))
	
	cases_predictions = {}
	cases_targets = {}
	cases_folds = {}
	for i in range(len(filenames)):
		pid = fname2pid(filenames[i])
		prev_pred = cases_predictions.get(pid, np.zeros(nb_categories))
		preds = predictions[i]
		cases_predictions[pid] = prev_pred + preds
		cases_targets[pid] = targets[i]
		cases_folds[pid] = pid_fold[i]
	
	print('{} cases'.format(len(cases_predictions)))
	
	y_pred = []
	y_true = []
	y_id = []
	y_fold = []
	for pid in cases_predictions:
		y_pred.append(cases_predictions[pid][0])
		y_true.append(cases_targets[pid])
		y_id.append(pid)
		y_fold.append(cases_folds[pid])
	
	with open('../results/data/predictions_cv.csv', 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['ID', 'Prediction', 'Cancer', 'Fold'])
		for pid, prediction, gt, f in zip(y_id, y_pred, y_true, y_fold):
			pid = pid.lstrip('0')
			csvwriter.writerow([pid, prediction, gt[0], f])
	
	fpr, tpr, thresholds = roc_curve(y_true, y_pred)
	roc_auc = roc_auc_score(y_true, y_pred)
	
	print('roc auc = {}'.format(roc_auc_score(y_true, y_pred)))
	
	plt.rcParams.update({'font.size': 24})
	
	fig = plt.figure(figsize=(10, 10))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
	plt.grid(color='silver', alpha=0.3, linestyle='--', linewidth=1)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig('roc_cv.png', bbox_inches='tight')
	plt.close(fig)


def test_train(folds):
	for f in range(folds):
		filenames, predictions, targets = predict(f, folds, test_cases=False)
		pid_fold = [f] * len(targets)
		
		cases_predictions = {}
		cases_targets = {}
		cases_folds = {}
		for i in range(len(filenames)):
			pid = fname2pid(filenames[i])
			prev_pred = cases_predictions.get(pid, np.zeros(nb_categories))
			preds = predictions[i]
			cases_predictions[pid] = prev_pred + preds
			cases_targets[pid] = targets[i]
			cases_folds[pid] = pid_fold[i]
		
		y_pred = []
		y_true = []
		y_id = []
		y_fold = []
		for pid in cases_predictions:
			y_pred.append(cases_predictions[pid][0])
			y_true.append(cases_targets[pid])
			y_id.append(pid)
			y_fold.append(cases_folds[pid])
		
		with open('../results/data/predictions_cv_train.csv', 'a') as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow(['ID', 'Prediction', 'Cancer', 'Fold'])
			for pid, prediction, gt, fo in zip(y_id, y_pred, y_true, y_fold):
				pid = pid.lstrip('0')
				csvwriter.writerow([pid, prediction, gt[0], fo])


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
		test(int(sys.argv[2]))
		test_train(int(sys.argv[2]))
