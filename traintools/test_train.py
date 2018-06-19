import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve

from data import train_data, train_pids
from focal_loss import focal_loss

checkpoints_dir = (
    "/media/maciej/Thyroid/thyroid-nodules/multitask/custom-test/checkpoints/"
)
batch_size = 128
nb_categories = 1


def predict():
    weights_path = os.path.join(checkpoints_dir, "weights.h5")

    net = load_model(weights_path, custom_objects={"focal_loss_fixed": focal_loss()})

    X_train, y_train = train_data()

    preds = net.predict(X_train, batch_size=batch_size, verbose=1)

    return preds[0], y_train["out_cancer"]


def test():
    predictions, targets = predict()

    cases_predictions = {}
    cases_targets = {}
    pids = train_pids()
    for i in range(len(pids)):
        pid = pids[i]
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

    with open("../results/data/predictions_train.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["ID", "Prediction", "Cancer"])
        for pid, prediction, gt in zip(y_id, y_pred, y_true):
            pid = pid.lstrip("0")
            csvwriter.writerow([pid, prediction, gt[0]])

    plot_roc(y_true, y_pred)


def plot_roc(y_true, y_pred, figname="roc_train.png"):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print("roc auc = {}".format(roc_auc))

    plt.rcParams.update({"font.size": 24})

    fig = plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    device = "/gpu:" + sys.argv[1]
    with tf.device(device):
        test()
