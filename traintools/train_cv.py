import os
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from sklearn.metrics import roc_auc_score

from data import fold_data, augment#, augment_2, augment_4
from model import multitask_cnn, loss_dict, loss_weights_dict
#from model_resnet import multitask_resnet, loss_dict, loss_weights_dict

checkpoints_dir = (
    "/home/adithya/Desktop/Adithya_Thyroid_Deep_Learning/ECHOGENECITY/deep-feature-extraction-threeclass/checkpoints/<FOLD>/"
)
logs_dir = "/home/adithya/Desktop/Adithya_Thyroid_Deep_Learning/ECHOGENECITY/deep-feature-extraction-threeclass/logs/<FOLD>/"

batch_size = 128
epochs = 250
base_lr = 0.001
nb_categories = 3


def train(fold):
    fold_checkpoints_dir = checkpoints_dir.replace("<FOLD>", str(fold))
    fold_logs_dir = logs_dir.replace("<FOLD>", str(fold))

    if not os.path.exists(fold_checkpoints_dir):
        os.makedirs(fold_checkpoints_dir)
    if not os.path.exists(fold_logs_dir):
        os.makedirs(fold_logs_dir)

    x_train, y_train, x_test, y_test = fold_data(fold, False)

    print("Training and validation data processed.")
    print("Training data shape: {}".format(len(x_train)))
    print("Test data shape: {}".format(len(x_test)))
    
    model = multitask_cnn(nb_categories)

    optimizer = RMSprop(lr=base_lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=["accuracy"],
    )

    training_log = TensorBoard(
        log_dir=os.path.join(fold_logs_dir, "log"), write_graph=False
    )

    callbacks = [training_log]
    
    print("Y test compos: ", y_test.shape)
    print("Y train compos: ", y_train.shape)


   # print("Y_TEST SHAPE: ", y_test_compos.shape)
    
    for e in range(epochs):

        x_train_augmented = augment(x_train)

	model.fit(x={"thyroid_input": x_train_augmented}, y=y_train,class_weight={0:1.0, 1:627.0/565.0, 2:627.0/27.0}, validation_data=(x_test, y_test), batch_size=batch_size, epochs=e + 1, initial_epoch=e, shuffle=True, callbacks=callbacks)


    model.save(os.path.join(fold_checkpoints_dir, "weights.h5"))

    print("Training fold {} completed.".format(fold))


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    train_fold = 0
    
    while(train_fold <=9):
	print("current train fold: ", train_fold)
        train(train_fold)
        train_fold = train_fold + 1
