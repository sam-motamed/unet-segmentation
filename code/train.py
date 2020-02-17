 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Sam Motamed
"""

from __future__ import division, print_function
import numpy as np
from keras import backend as K
import pydicom as dicom
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.exposure import equalize_hist
import scipy.ndimage as nd
from skimage import util
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.callbacks import LearningRateScheduler, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from skimage.segmentation import chan_vese
import tensorflow as tf
from models import actual_unet, simple_unet, N2
#from metrics import dice_coef, dice_coef_loss

#from augmenters import elastic_transform

#uncomment to print full npo matrice content fro debugging purposes
#np.set_printoptions(threshold=np.inf)
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def dicom_to_array(img_rows, img_cols):
    track_ptn = []
    for direc in ['uhn_train', 'uhn_test', 'uhn_val']:
        print(direc)
        PathDicom = 'C:/Users/mshri/Desktop/prostate_segmentation-master/data/' + direc + '/'
        pathMask = 'C:/Users/mshri/Desktop/prostate_segmentation-master/data/python_masks/'
        dcm_dict = dict()  # create an empty list
        for dirName, subdirList, fileList in os.walk(PathDicom):
            if ('DWI' in dirName):

                imgs_all = []
                if any(".dcm" in s for s in fileList):
                    ptn_name = (dirName.split('/') [-1]).split('\\')[0]
                    ptn_name_pure = ptn_name.split('_')[1]
                    if not ptn_name in track_ptn:
                        #for modal in ['B0','B1K', 'B1600','B400']:
                            
                        track_ptn.append(ptn_name)
                        #print(track_ptn)
                        path_ptn = os.path.join(PathDicom, ptn_name, 'DWI')
                        #print(modal, path_ptn)
                        fileList = list(filter(lambda x: '.dcm' in x, os.listdir(path_ptn)))
                        #print(fileList)
                        indice = np.sort([ int( fname[:-4] ) for fname in fileList])
                        end_len = indice[-1]
                        imgs = np.zeros([int(indice[-1] / 4), img_rows, img_cols])
                        for filename in np.sort(fileList):

                            if int (filename[:-4]) <= int(indice[-1] / 4):
                                img = dicom.read_file(os.path.join(dirName,filename)).pixel_array
                                img = equalize_hist( img.astype(int) )
                                #img = img / 255.
                                #img *= 255.0/img.max()
                                #img = equalize_hist( img.astype(int))
                                #print("image file name", filename, "shape of image", img.shape)
                                #img = crop_center(img,200,200)
                                img = resize(img, (img_rows, img_cols), preserve_range=True)
        
                                imgs[int(filename[:-4]) - 1] = img
                        imgs_all.append(imgs)

                        #imgs[int(filename[:-4]) - 1 + end_len] = cv
                else:
                    pass
                
                if len(imgs_all) > 0:
                    xs = ((imgs_all[0].shape)[0])
                    imgs_all_array = np.zeros([xs, img_rows, img_cols])
                    for i in range(len(imgs_all)):
                        for j in range(len(imgs_all[i])):
                            imgs_all_array[(i * xs) + j] = imgs_all[i][j]
                    dcm_dict[ptn_name] = imgs_all_array
        
        
        
        
        
        
        
        track_ptn_mask = []
        pathMask = 'C:/Users/mshri/Desktop/prostate_segmentation-master/data/'+ direc + '_mask/'
        dcm_dict_mask = dict()  # create an empty list
        for dirName, subdirList, fileList in os.walk(pathMask):
            if ('DWI' in dirName and 'TZ' in dirName or 'Tz' in dirName or 'tZ' in dirName or 'tz' in dirName or 'TC' in dirName):
            #if ('DWI' in dirName and 'Prostate' in dirName):
                if 'TZ' in dirName:
                    mod_name = 'TZ'
                elif 'tZ' in dirName:
                    mod_name = 'tZ'
                elif 'tz' in dirName:
                    mod_name = 'tz'
                elif 'TC' in dirName:
                    mod_name = 'TC'
                else:
                    mod_name = 'Tz'
                masks_all = []
                if any(".dcm" in s for s in fileList):
                    ptn_name = (dirName.split('/') [-1]).split('\\')[0]
                    ptn_name_pure = ptn_name.split('_')[1]
                    if not ptn_name in track_ptn_mask:
                        #for modal in ['B0','B1K', 'B1600','B400']:
                            
                        track_ptn_mask.append(ptn_name)
                        #print(track_ptn)
                        #print(dirName.split('\\'))
                        modal = dirName.split('\\')[-2]
                        path_ptn = os.path.join(pathMask, ptn_name, modal, mod_name)
                        fileList = list(filter(lambda x: '.dcm' in x, os.listdir(path_ptn)))
                        #print(fileList)
                        indice = np.sort([ int( fname[4:-4] ) for fname in fileList])
                        end_len = indice[-1]
                        imgs = np.zeros([int(indice[-1]/4), img_rows, img_cols])
                        for filename in np.sort(fileList):
                            if int(filename[4:-4]) <= int(indice[-1]/4):
                                img_mask = dicom.read_file(os.path.join(dirName,filename)).pixel_array
                                img_mask.astype(int) 
                                img_mask.setflags(write=1)
                                img_mask = img_mask /  255.
                                #img = crop_center(img,200,200)
                                img_mask = resize(img_mask, (img_rows, img_cols), preserve_range=True)

                                imgs[int(filename[4:-4]) - 1] = img_mask
                        num_slices = np.shape(imgs)[0]
        
                        imgs_copy = np.zeros(imgs.shape)
                        im_im = np.vstack((imgs[:int(num_slices / 4)], imgs[:int(num_slices / 4)]))
                        imgs_copy = np.vstack((im_im, im_im))
                        masks_all.append(imgs)
                        
                else:
                    pass
                
                if len(masks_all) > 0:
                    xs = ((masks_all[0].shape)[0])
                    masks_all_array = np.zeros([xs, img_rows, img_cols])
                    for i in range(len(masks_all)):
                        for j in range(len(masks_all[i])):
                            masks_all_array[(i * xs) + j] = masks_all[i][j]
                    dcm_dict_mask[ptn_name] = masks_all_array
        
        imgs= []
        img_masks = []    
        for patient in dcm_dict.keys():
            mask = dcm_dict_mask[patient]
            img = dcm_dict[patient]
            #print(patient)
            if len(img) != len(mask) :
                print('Dimension mismatch for', patient, 'in folder', direc)
                print("image: ", len(img), "-----", "mask: ", len(mask))
            else:
                img_masks.append(mask)
                imgs.append(img)
        imgs = np.concatenate(imgs, axis=0).reshape(-1, img_rows, img_cols, 1)
        img_masks = np.concatenate(img_masks, axis=0).reshape(-1, img_rows, img_cols, 1)
        
        print(imgs.shape)
        print(img_masks.shape)
            
        #I will do only binary classification for now
        img_masks = np.array(img_masks > 0, dtype=int)

        #img_masks = np.array(img_masks>0.45, dtype=int)
        np.save('C:/Users/mshri/Desktop/prostate_segmentation-master/data/' + direc + '.npy', imgs)
        np.save('C:/Users/mshri/Desktop/prostate_segmentation-master/data/' + direc + '_mask.npy', img_masks)
smooth = 1.
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
def dice_coef(y_true, y_pred):

    tf.cast(y_pred, tf.int32)
    #tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    intersect = K.sum(y_pred * y_true)

    denominator = K.sum(y_pred_f) + K.sum(y_true_f)
    dice_score = K.constant(2.) * intersect / (denominator)
    result = tf.keras.backend.switch(
    K.equal(denominator, 0.),
    1.,
    dice_score
)
    return result
    
    
def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)    


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_data():

    X_train = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_train.npy')
    y_train = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_train_mask.npy')
    X_val = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_val.npy')
    y_val = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_val_mask.npy')
    X_test = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_test.npy')
    y_test = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_test_mask.npy')
    return X_train, y_train, X_test, y_test, X_val, y_val
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.05
    epochs_drop = 5
    lrate = initial_lrate * drop**int((1 + epoch) / epochs_drop)
    return lrate

def keras_fit_generator(img_rows=144, img_cols=144, n_imgs=10**4 , batch_size=32, regenerate=True):
    if regenerate:
        dicom_to_array(img_rows, img_cols)
        print(img_rows, img_cols)
        #preprocess_data()

    X_train, y_train, X_test, y_test, X_val, y_val = load_data()
    img_rows = X_train.shape[1]
    img_cols = img_rows
    
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2)#,
        #preprocessing_function=elastic_transform)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(X_train,seed=seed)
    mask_datagen.fit(y_train, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)


    model = N2(img_rows, img_cols)
    
    '''model.load_weights('C:/Users/mshri/Desktop/MEAN_SHIFT_DWI/weights.h5')
    for i,layer in enumerate(model.layers):
        print(i,layer.name)
    for layer in model.layers:
        layer.trainable=False
    for layer in model.layers[:50]:
        layer.trainable=True
    for layer in model.layers[100:]:
        layer. trainable=True'''
        
        
    model_checkpoint = ModelCheckpoint(
        'C:/Users/mshri/Desktop/prostate_segmentation-master/data/weights.h5', monitor='val_loss',save_best_only = True, save_weights_only=True)
    csvLog=CSVLogger('C:/Users/mshri/Desktop/prostate_segmentation-master/data/logs.csv', separator=',', append=True)
    lrate = LearningRateScheduler(step_decay)

    model.compile(optimizer=Adam(),loss=dice_coef_loss, metrics=[dice_coef])
   
    
    model.fit(X_train, y_train, batch_size=32, nb_epoch=25, verbose=1, shuffle=True,
              validation_data=(X_test, y_test),
              callbacks=[model_checkpoint])

import time

start = time.time()
keras_fit_generator(img_rows=144, img_cols=144, regenerate=False ,n_imgs=30*10**4, batch_size=32)

end = time.time()

print('Elapsed time:', round((end-start)/60, 2 ) )
