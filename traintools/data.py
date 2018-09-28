from glob import glob
from random import seed, randint

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from imgaug import augmenters
from scipy.misc import imread

data_path = "./data.csv"
images_dir = "/home/adithya/Desktop/Adithya_Thyroid_Deep_Learning/data/Nodules-originals"
mask_dir = "/home/adithya/Desktop/Adithya_Thyroid_Deep_Learning/data/Nodule-masks"

#DONT NEED THIS: test_images_dir = "/media/maciej/Thyroid/thyroid-nodules/images-test"

random_seed = 3
total_folds = 10


def fold_pids(fold, test=True):
    df = pd.read_csv(data_path)
    all_files = glob(images_dir + "/*.PNG")
    val_ids = validation_ids(fold, df[["ID", "Cancer"]])

    pids = []

    for f_path in all_files:
        pid = fname2pid(f_path)
        if (test and pid in val_ids) or (not test and pid not in val_ids):
            pids.append(pid)

    return pids


def test_pids():
    test_files = sorted(glob(test_images_dir + "/*.PNG"))
    pids = []
    
    for f_path in test_files:
        pids.append(fname2pid(f_path))
    
    return pids


def train_pids():
    train_files = sorted(glob(images_dir + "/*.PNG"))
    pids = []
    
    for f_path in train_files:
        pids.append(fname2pid(f_path))
        
    return pids


def train_data():
    df = pd.read_csv(data_path)
    df.fillna(0, inplace=True)
    df.Calcs1.replace(0, "None", inplace=True)

    df_cancer = df[["ID", "Cancer"]]
    df_compos = pd.concat([df.ID, pd.get_dummies(df.Composition)], axis=1)
    df_echo = pd.concat([df.ID, pd.get_dummies(df.Echogenicity)], axis=1)
    df_shape = df[["ID", "Shape"]]
    df_shape["Shape"] = df_shape.apply(lambda row: 1 if row.Shape == "y" else 0, axis=1)
    df_calcs = pd.concat([df.ID, pd.get_dummies(df.Calcs1)], axis=1)
    df_margin = pd.concat([df.ID, pd.get_dummies(df.MargA)], axis=1)

    train_files = sorted(glob(images_dir + "/*.PNG"))
    X_train = []

    y_train_cancer = []
    y_train_compos = []
    y_train_echo = []
    y_train_shape = []
    y_train_calcs = []
    y_train_margin = []

    for f_path in train_files:
        pid = fname2pid(f_path)
        X_train.append(
            np.expand_dims(
                np.array(imread(f_path, flatten=False, mode="F")).astype(np.float32),
                axis=-1,
            )
        )
        y_train_cancer.append(
            df_cancer[df_cancer.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        y_train_compos.append(
            df_compos[df_compos.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        y_train_echo.append(
            df_echo[df_echo.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        if "trans" in f_path:
            y_train_shape.append(
                df_shape[df_shape.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
        else:
            y_train_shape.append(np.array([0]).astype(np.float32))
        y_train_calcs.append(
            df_calcs[df_calcs.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        y_train_margin.append(
            df_margin[df_margin.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )

    X_train = np.array(X_train)

    X_train /= 255.
    X_train -= 0.5
    X_train *= 2.

    y_train = {
        "out_cancer": np.array(y_train_cancer),
        "out_compos": np.array(y_train_compos),
        "out_echo": np.array(y_train_echo),
        "out_shape": np.array(y_train_shape),
        "out_calcs": np.array(y_train_calcs),
        "out_margin": np.array(y_train_margin),
    }

    return X_train, y_train


def test_data():
    df = pd.read_csv(data_path)
    df.fillna(0, inplace=True)
    df.Calcs1.replace(0, "None", inplace=True)

    df_cancer = df[["ID", "Cancer"]]
    df_compos = pd.concat([df.ID, pd.get_dummies(df.Composition)], axis=1)
    df_echo = pd.concat([df.ID, pd.get_dummies(df.Echogenicity)], axis=1)
    df_shape = df[["ID", "Shape"]]
    df_shape["Shape"] = df_shape.apply(lambda row: 1 if row.Shape == "y" else 0, axis=1)
    df_calcs = pd.concat([df.ID, pd.get_dummies(df.Calcs1)], axis=1)
    df_margin = pd.concat([df.ID, pd.get_dummies(df.MargA)], axis=1)

    test_files = sorted(glob(test_images_dir + "/*.PNG"))

    X_test = []

    y_test_cancer = []
    y_test_compos = []
    y_test_echo = []
    y_test_shape = []
    y_test_calcs = []
    y_test_margin = []

    for f_path in test_files:
        pid = fname2pid(f_path)
        X_test.append(
            np.expand_dims(
                np.array(imread(f_path, flatten=False, mode="F")).astype(np.float32),
                axis=-1,
            )
        )
        y_test_cancer.append(
            df_cancer[df_cancer.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        y_test_compos.append(
            df_compos[df_compos.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        y_test_echo.append(
            df_echo[df_echo.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        if "trans" in f_path:
            y_test_shape.append(
                df_shape[df_shape.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
        else:
            y_test_shape.append(np.array([0]).astype(np.float32))
        y_test_calcs.append(
            df_calcs[df_calcs.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )
        y_test_margin.append(
            df_margin[df_margin.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
        )

    X_test = np.array(X_test)

    X_test /= 255.
    X_test -= 0.5
    X_test *= 2.

    y_test = [
        np.array(y_test_cancer),
        np.array(y_test_compos),
        np.array(y_test_echo),
        np.array(y_test_shape),
        np.array(y_test_calcs),
        np.array(y_test_margin),
    ]

    return X_test, y_test


def fold_data(fold):
    df = pd.read_csv(data_path)
    df.fillna(0, inplace=True)
    df.Calcs1.replace(0, "None", inplace=True)

    df_cancer = df[["ID", "Cancer"]]
    df_compos = pd.concat([df.ID, pd.get_dummies(df.Composition)], axis=1)
    df_echo = pd.concat([df.ID, pd.get_dummies(df.Echogenicity)], axis=1)
    df_shape = df[["ID", "Shape"]]
    df_shape["Shape"] = df_shape.apply(lambda row: 1 if row.Shape == "y" else 0, axis=1)
    df_calcs = pd.concat([df.ID, pd.get_dummies(df.Calcs1)], axis=1)
    df_margin = pd.concat([df.ID, pd.get_dummies(df.MargA)], axis=1)

    all_files = glob(images_dir + "/*.PNG")
    all_masks = glob(mask_dir + "/*.PNG")
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

    for f_path, m_path in zip(all_files, all_masks):
        pid = fname2pid(f_path)
        image = np.expand_dims(
            np.array(imread(f_path, flatten=False, mode="F")).astype(np.float32),
            axis=-1,
        )
       # print("og im shape: ", image.shape)
        mask = np.expand_dims(
            np.array(imread(m_path, flatten=False, mode="F")).astype(np.float32),
            axis=-1,
        )
	
	#print("mask shape: ", mask.shape)

        image = np.append(image, mask, axis=2)
	#print("new im shape: ", image.shape)
        if pid in val_ids:
            X_test.append(image)
            y_test_cancer.append(
                df_cancer[df_cancer.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
            y_test_compos.append(
                df_compos[df_compos.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
            y_test_echo.append(
                df_echo[df_echo.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
            )
            if "trans" in f_path:
                y_test_shape.append(
                    df_shape[df_shape.ID == pid]
                    .as_matrix()
                    .flatten()[1:]
                    .astype(np.float32)
                )
            else:
                y_test_shape.append(np.array([0]).astype(np.float32))
            y_test_calcs.append(
                df_calcs[df_calcs.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
            y_test_margin.append(
                df_margin[df_margin.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
        else:
            X_train.append(image)
            y_train_cancer.append(
                df_cancer[df_cancer.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
            y_train_compos.append(
                df_compos[df_compos.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
            y_train_echo.append(
                df_echo[df_echo.ID == pid].as_matrix().flatten()[1:].astype(np.float32)
            )
            if "trans" in f_path:
                y_train_shape.append(
                    df_shape[df_shape.ID == pid]
                    .as_matrix()
                    .flatten()[1:]
                    .astype(np.float32)
                )
            else:
                y_train_shape.append(np.array([0]).astype(np.float32))
            y_train_calcs.append(
                df_calcs[df_calcs.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )
            y_train_margin.append(
                df_margin[df_margin.ID == pid]
                .as_matrix()
                .flatten()[1:]
                .astype(np.float32)
            )

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train /= 255.
    X_train -= 0.5
    X_train *= 2.

    X_test /= 255.
    X_test -= 0.5
    X_test *= 2.

    y_train = {
        "out_cancer": np.array(y_train_cancer),
        "out_compos": np.array(y_train_compos),
        "out_echo": np.array(y_train_echo),
        "out_shape": np.array(y_train_shape),
        "out_calcs": np.array(y_train_calcs),
        "out_margin": np.array(y_train_margin),
    }

    y_test = [
        np.array(y_test_cancer),
        np.array(y_test_compos),
        np.array(y_test_echo),
        np.array(y_test_shape),
        np.array(y_test_calcs),
        np.array(y_test_margin),
    ]

    return X_train, y_train, X_test, y_test


def augment(X):
    seq = augmenters.Sequential(
        [
            augmenters.Fliplr(0.5),
            augmenters.Flipud(0.5),
            augmenters.Affine(rotate=(-15, 15)),
            augmenters.Affine(shear=(-15, 15)),
            augmenters.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            augmenters.Affine(scale=(0.9, 1.1)),
        ]
    )

    return seq.augment_images(X)


#TODO: will have to get rid of the last few lines in train which normalize the entire training set at once, since we will be doing that here (JK --> im not sure if we r doin it here or not, it will depend ...)
def augment_2(img, mask):
   
    #print("img og shape:", img.shape)
    img = np.expand_dims(img, axis=2)
    #print("img new shape:", img.shape) 
    
    #Start off with an elastic transformation on the nodule image
    img = tl.prepro.elastic_transform(img, alpha=img.shape[1]*3, sigma=img.shape[1]*0.07)
	
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.rgb_to_hsv(img)
    
    og_img_shape = img.shape
    og_mask_shape = mask.shape

    print("mask shape!", mask.shape)

    #Randomly flip the image
    r_flip = tf.random_uniform([3], 0, 1.0, dtype=tf.float32)

    #Left right
    mirror = tf.less(r_flip[0], 0.5)
    with tf.Session() as default:
	mirror = mirror.eval()
    
    if(mirror):
	img = tf.reverse(img, tf.stack([1]))
	mask = tf.reverse(mask, tf.stack([1]))    
    
    #Up down
    mirror = tf.less(r_flip[1], 0.5)
    with tf.Session() as default:
	mirror = mirror.eval()

    if(mirror):
	img = tf.reverse(img, tf.stack([0]))
    	mask = tf.reverse(mask, tf.stack([0]))    

    #Transpose
    mirror = tf.less(tf.stack([r_flip[2], 1.0 - r_flip[2]]), 0.5)
    mirror = tf.cast(mirror, tf.int32)
    mirror = tf.stack([mirror[0], mirror[1], 2])
    mirror_mask = tf.stack([mirror[0], mirror[1]])

    img = tf.transpose(img, perm=mirror) 
    mask = tf.transpose(mask, perm=mirror_mask)

    #Adjust the image so that it's back to its original shape after the transpose operation
    img.set_shape(og_img_shape)
   
    print("og_mask_shape: ", og_mask_shape)
    mask.set_shape(og_mask_shape)

   #Contrast and saturation
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)

    #Return image to original dimensions as a grayscale
    img = tf.image.hsv_to_rgb(img)
    img = tf.image.rgb_to_grayscale(img)

    with tf.Session() as default:
        img = img.eval()
        mask = mask.eval()
  
    img = img.astype(np.int8)
    mask = mask.astype(np.int8)

    return np.squeeze(img), np.squeeze(mask)

#assume X is a numpy tensor of dimensions n x 160 x 160 x 2
def augment_4(X):
    print("X.shape:", X.shape)
    n_imgs = X.shape[0]

    X_augmented = np.copy(X)
    
    for i in range(n_imgs):
	current_im = X[i,:,:,0]
	current_mask = X[i,:,:,1]

    current_im, current_mask = augment_2(current_im, current_mask)
 
    X_augmented[i,:,:,0] = current_im
    X_augmented[i,:,:,1] = current_mask
    
    return X_augmented



def validation_ids(fold, df_cancer):
    pid_set = set()
    all_files = glob(images_dir + "/*.PNG")
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
    return fname.split("/")[-1].split(".")[0].lstrip("0")
