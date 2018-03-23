import csv
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from scipy.misc import imread
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from focal_loss import focal_loss

data_path = './data.csv'
images_dir = '/media/maciej/Thyroid/thyroid-nodules/images-test'
epoch = 250
checkpoints_dir = '/media/maciej/Thyroid/thyroid-nodules/multitask/custom-test/checkpoints/'
batch_size = 128
nb_categories = 1


def predict():
	weights_path = os.path.join(checkpoints_dir, 'weights_{}.h5'.format(epoch))
	
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
	
	test_files = glob(images_dir + '/*.PNG')
	
	X_test = []
	
	y_test_cancer = []
	y_test_compos = []
	y_test_echo = []
	y_test_shape = []
	y_test_calcs = []
	y_test_margin = []
	
	for f_path in tqdm(test_files):
		pid = fname2pid(f_path)
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


def test():
	filenames, predictions, targets = predict()
	
	print(len(filenames))
	
	cases_predictions = {}
	cases_targets = {}
	for i in range(len(filenames)):
		pid = fname2pid(filenames[i])
		prev_pred = cases_predictions.get(pid, np.zeros(nb_categories))
		preds = predictions[i]
		cases_predictions[pid] = prev_pred + preds
		cases_targets[pid] = targets[i]
	
	y_pred = []
	y_true = []
	y_id = []
	for pid in cases_predictions:
		y_pred.append(cases_predictions[pid][0])
		y_true.append(cases_targets[pid])
		y_id.append(pid)
	
	with open('../results/data/predictions_test.csv', 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['ID', 'Prediction', 'Cancer'])
		for pid, prediction, gt in zip(y_id, y_pred, y_true):
			pid = pid.lstrip('0')
			csvwriter.writerow([pid, prediction, gt[0]])
	
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
	plt.savefig('roc_test.png', bbox_inches='tight')
	plt.close(fig)


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
		test()
