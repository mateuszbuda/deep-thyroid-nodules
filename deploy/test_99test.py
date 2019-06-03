import csv
import os
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

from data import test_data
from focal_loss import focal_loss

PATH_TO_WEIGHTS = "./weights.h5"
batch_size = 128
nb_categories = 1


def predict():
    weights_path = os.path.join(PATH_TO_WEIGHTS)

    net = load_model(weights_path, custom_objects={"focal_loss_fixed": focal_loss()})

    X_test, pids = test_data()

    preds = net.predict(X_test, batch_size=batch_size, verbose=1)

    return preds[0], pids


def test():
    predictions, pids = predict()

    cases_predictions = {}
    for i in range(len(pids)):
        pid = pids[i]
        prev_pred = cases_predictions.get(pid, np.zeros(nb_categories))
        preds = predictions[i]
        cases_predictions[pid] = prev_pred + preds

    y_pred = []
    y_id = []
    for pid in cases_predictions:
        y_pred.append(cases_predictions[pid][0])
        y_id.append(pid)

    with open("/data/predictions_test.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["ID", "Prediction"])
        for pid, prediction in zip(y_id, y_pred):
            pid = pid.lstrip("0")
            csvwriter.writerow([pid, prediction])


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    device = "/gpu:" + sys.argv[1]
    with tf.device(device):
        test()
