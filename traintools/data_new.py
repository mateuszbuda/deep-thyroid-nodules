from glob import glob
from random import seed, randint

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from imgaug import augmenters
from scipy.misc import imread
from sklearn.model_selection import StratifiedKFold

#data_path_og = "./data.csv"
data_path = "/.data.csv" #"/home/adithya/Desktop/Adithya_Thyroid_Deep_Learning/radiologist_data/cleaned_labels/relabelled_composition.csv"

images_dir = "/home/adithya/Desktop/Adithya_Thyroid_Deep_Learning/data/Nodules-originals"
mask_dir = "/home/adithya/Desktop/Adithya_Thyroid_Deep_Learning/data/Nodule-masks"



total_folds = 10


#TODO: balance folds
def fold_data(fold):
    df = pd.read_csv(data_path)
    df.fillna(0, inplace=True)

    ids = df["ID"]
    df_echo = df["Echogenecity"]


    label_dict = {
       'Cant classify': -1,
       'Hyper': 0,
       'Iso': 0,
       'Mild Hypo': 1,
       'Very Hypo': 2
    }


    labels = list()
    for label in df_compos:
	labels.append(label_dict[label])
   
    ids = ids.values.tolist()


    print("YEET")
    print(len(labels), len(ids))
    

    all_labels_dict = dict(zip(ids, labels))

   
    all_files = glob(images_dir + "/*.PNG")
    all_masks = glob(mask_dir + "/*.PNG")


    X = []
    y = []
    

    for f_path, m_path in zip(all_files, all_masks):
        pid = fname2pid(f_path)

        if pid in ids:
        	image = np.expand_dims(np.array(imread(f_path, flatten=False, mode="F")).astype(np.float32),axis=-1)
        	mask = np.expand_dims(np.array(imread(m_path, flatten=False, mode="F")).astype(np.float32),axis=-1)
        	image = np.append(image, mask, axis=2)


		X.append(image)
		y.append(all_labels_dict[pid])

    X = np.array(X)
    X /= 255.
    X -= 0.5
    X *= 2.

    y = np.array(y)
    skf = StratifiedKFold(n_splits=total_folds, random_state=5)

    splits = skf.split(X,y)

    i = 0
    fold_pids = []
    for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	if(i == fold):
		y_train = get_one_hot(y_train, 3)
		y_test = get_one_hot(y_test, 3)
		fold_pids = np.array(ids)[test_index]
                
		return X_train, y_train, X_test, y_test, fold_pids

        i += 1




def fold_data2(fold):
    df = pd.read_csv(data_path)
    df.fillna(0, inplace=True)


    df_og = pd.read_csv(data_path_og)
    df_og.fillna(0, inplace=True)

    ids = df["ID"]
    df_compos = df["Composition"]

    ids_og = df_og["ID"]
    df_compos_og = df_og["Composition"]

    label_dict = {
       'Cystic': 0,
       'Spongiform': 0,
       'Mixed cystic_solid': 1,
       'Entirely or almost entirely solid': 2
    }


    labels = list()
    for label in df_compos:
	labels.append(label_dict[label])
   
    ids = ids.values.tolist()


    all_labels_dict = dict(zip(ids, labels))



    labels_og = list()
    for label in df_compos_og:
        if label in label_dict.keys():
		labels_og.append(label_dict[label])
   
    ids_og = ids_og.values.tolist()


    all_labels_dict_og = dict(zip(ids_og, labels_og))   


    all_files = glob(images_dir + "/*.PNG")
    all_masks = glob(mask_dir + "/*.PNG")


    X = []
    y = []
    

    cnt = 0
    for f_path, m_path in zip(all_files, all_masks):
        pid = fname2pid(f_path)

	
        if pid in ids:
        	image = np.expand_dims(np.array(imread(f_path, flatten=False, mode="F")).astype(np.float32),axis=-1)
        	mask = np.expand_dims(np.array(imread(m_path, flatten=False, mode="F")).astype(np.float32),axis=-1)
        	image = np.append(image, mask, axis=2)

                
		X.append(image)
               
                start_index = 0
	        end_index = -5
                try:
                  
			while(pid[start_index] == '0'): start_index += 1
			
		
			if('long' in pid):
                        	end_index = -4
                        
                        if(pid[end_index-1] == '_'): end_index += 1
                        #print("original pid: ", pid, "transformed pid: ", pid[start_index:end_index])
			y.append(all_labels_dict_og[pid[start_index:end_index]])
		except:
                        print("Couldnt find stupid pid: ", pid[start_index:end_index])
			y.append(all_labels_dict[pid])
			cnt += 1
	
    print("CNT: ", cnt)
    X = np.array(X)
    X /= 255.
    X -= 0.5
    X *= 2.

    y = np.array(y)
    skf = StratifiedKFold(n_splits=total_folds, random_state=5)

    splits = skf.split(X,y)

    i = 0

    fold_pids = []

    for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
        	
	if(i == fold):
		y_train = get_one_hot(y_train, 3)
		y_test = get_one_hot(y_test, 3)
                fold_pids = np.array(ids)[test_index]
                print(len(fold_pids),"Fold pids: ", fold_pids[1:10])        
		return X_train, y_train, X_test, y_test, fold_pids

        i += 1





def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])




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



def fname2pid(fname):
    return fname.split("/")[-1].split(".")[0].lstrip("0")


	
#def fname2pid(fname):
 #   return fname.split("/")[-1].split(".")[0] + fname.split("/")[-1].split(".")[1]

fold_data2(5)
