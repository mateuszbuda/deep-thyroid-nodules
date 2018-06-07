import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve

from data import data, fold_pids
from focal_loss import focal_loss

checkpoints_dir = (
    "/media/maciej/Thyroid/thyroid-nodules/multitask/custom/checkpoints/<FOLD>/"
)
weights_file = "weights_250.h5"
batch_size = 128
nb_categories = 1


def predict(fold, test_cases=True):
    fold_checkpoints_dir = checkpoints_dir.replace("<FOLD>", str(fold))
    weights_path = os.path.join(fold_checkpoints_dir, weights_file)

    net = load_model(weights_path, custom_objects={"focal_loss_fixed": focal_loss()})

    x_train, y_train, x_test, y_test = data(fold)

    if test_cases:
        preds = net.predict(x_test, batch_size=batch_size, verbose=1)
        y = y_test[0]
    else:
        preds = net.predict(x_train, batch_size=batch_size, verbose=1)
        y = y_train["out_cancer"]

    return preds[0], y


def test(folds):
    pids = []
    predictions = np.zeros((0, nb_categories))
    targets = []
    pid_fold = []

    for f in range(folds):
        preds, t = predict(f, folds)
        predictions = np.vstack((predictions, preds))
        pids.extend(fold_pids(f))
        targets.extend(t)
        pid_fold.extend([f] * len(t))

    print("{} images".format(len(pids)))

    cases_predictions = {}
    cases_targets = {}
    cases_folds = {}
    for i in range(len(pids)):
        pid = pids[i]
        prev_pred = cases_predictions.get(pid, np.zeros(nb_categories))
        preds = predictions[i]
        cases_predictions[pid] = prev_pred + preds
        cases_targets[pid] = targets[i]
        cases_folds[pid] = pid_fold[i]

    print("{} cases".format(len(cases_predictions)))

    y_pred = []
    y_true = []
    y_id = []
    y_fold = []
    for pid in cases_predictions:
        y_pred.append(cases_predictions[pid][0])
        y_true.append(cases_targets[pid])
        y_id.append(pid)
        y_fold.append(cases_folds[pid])

    with open("../results/data/predictions_cv.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["ID", "Prediction", "Cancer", "Fold"])
        for pid, prediction, gt, f in zip(y_id, y_pred, y_true, y_fold):
            pid = pid.lstrip("0")
            csvwriter.writerow([pid, prediction, gt[0], f])

    plot_roc(y_true, y_pred)


def plot_roc(y_true, y_pred, figname="roc_cv.png"):
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


def test_train(folds):
    all_train_pred = []
    all_train_true = []

    for f in range(folds):
        predictions, targets = predict(f, test_cases=False)
        pid_fold = [f] * len(targets)
        pids = fold_pids(f, test=False)

        cases_predictions = {}
        cases_targets = {}
        cases_folds = {}
        for i in range(len(pids)):
            pid = pids[i]
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

        all_train_pred.extend(y_pred)
        all_train_true.extend([y[0] for y in y_true])

        with open("../results/data/predictions_cv_train.csv", "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            if f == 0:
                csvwriter.writerow(["ID", "Prediction", "Cancer", "Fold"])
            for pid, prediction, gt, fo in zip(y_id, y_pred, y_true, y_fold):
                pid = pid.lstrip("0")
                csvwriter.writerow([pid, prediction, gt[0], fo])

    plot_roc(all_train_true, all_train_pred, figname="roc_train.png")


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    device = "/gpu:" + sys.argv[1]
    with tf.device(device):
        test(int(sys.argv[2]))
        test_train(int(sys.argv[2]))
