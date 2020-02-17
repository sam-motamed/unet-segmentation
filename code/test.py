                        #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Sam Motamed

"""

from __future__ import division, print_function
from matplotlib import pyplot as plt
import numpy as np
print()
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.optimizers import Adam
from keras import backend as K
from models import actual_unet, simple_unet, N2
import os
import cv2
from statistics import mean, median
import matplotlib.gridspec as gridspec
import tensorflow as tf
import PIL
from PIL import Image, ImageTk
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
    0.,
    dice_score
)
    return result
    
def numpy_dice(y_true, y_pred):
    intersection = y_true*y_pred

    return ( 2. * intersection.sum())/ (np.sum(y_true) + np.sum(y_pred))
    


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def make_plots(img, segm, segm_pred):
    n_cols=3
    n_rows = len(img)

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[n_cols*mm])
        ax.imshow(img[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('MRI image', fontsize=20)
        ax = fig.add_subplot(gs[n_cols*mm+1])
        ax.imshow(segm[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('True Mask', fontsize=20)

        ax = fig.add_subplot(gs[n_cols*mm+2])
        ax.imshow(segm_pred[mm], cmap='gray' )
        ax.axis("image")  # remove axis
        ax.set_aspect(1)  # aspect ratio of 1
        if mm==0:
            ax.set_title('Predicted Mask', fontsize=20)
    return fig

smooth = 1.
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def check_predictions(n_best=5, n_worst=5):
    if not os.path.isdir('C:/Users/mshri/Desktop/prostate_segmentation-master/images'):
        os.mkdir('C:/Users/mshri/Desktop/prostate_segmentation-master/images')
        
    if not os.path.isdir('C:/Users/mshri/Desktop/prostate_segmentation-master/predicted_masks'):
        os.mkdir('C:/Users/mshri/Desktop/prostate_segmentation-master/predicted_masks')
        
    X_test = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_val.npy')
    #print("this is the dimension of x_test", X_test[1].shape)
    y_test = np.load('C:/Users/mshri/Desktop/prostate_segmentation-master/data/uhn_val_mask.npy')
    #store slice locations that do not have a mask in a list
    keep_record_nomask = []
    img_rows = X_test.shape[1]
    img_cols = img_rows
    num_iter = X_test.shape[0]

    for i in range(num_iter):
        #print(y_test[i].flatten().sum())
        if y_test[i].flatten().sum() == 0:
            keep_record_nomask.append(i)
    #print(keep_record_nomask)
    
    new_size = num_iter - len(keep_record_nomask)
    
    #print(new_size)
    new_X_test = np.zeros((new_size, img_rows, img_cols, 1))
    new_y_test = np.zeros((new_size, img_rows, img_cols, 1))
    #print(new_X_test.shape)
    for i in range(new_size):
        if i in keep_record_nomask:
             j = i
             while j in keep_record_nomask:
                 j += 1
             keep_record_nomask.append(j)
        else:
            j = i
            keep_record_nomask.append(j)
        #print(X_test[j].shape)
        new_X_test[i] = X_test[j]
        new_y_test[i]= y_test[j]
    #print(new_X_test.shape)        
    model = N2(img_rows, img_cols)
    model.load_weights('C:/Users/mshri/Desktop/prostate_segmentation-master/data/weights.h5')
    #model.load_weights('C:/Users/mshri/Desktop/MEAN_SHIFT_DWI/weights.h5')
    model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])

    #############################################################
    y_pred = model.predict(X_test, verbose=1)
    #y_test.astype(int)
    y_pred = y_pred > 0.1
    print(y_pred.shape)
    y_pred = y_pred.astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    
    
    #y_pred.astype(int)
    for i in range(1, y_pred.shape[0] - 1):
        y_pred[i] = cv2.morphologyEx(y_pred[i].reshape(144, 144), cv2.MORPH_CLOSE, kernel).reshape(144, 144, 1)
        y_pred[i] = cv2.morphologyEx(y_pred[i].reshape(144, 144), cv2.MORPH_OPEN, kernel).reshape(144, 144, 1)
        
        if np.count_nonzero(y_pred[i - 1]) > 0 and  np.count_nonzero(y_pred[i]) == 0 and np.count_nonzero(y_pred[i + 1]) > 0:
            y_pred[i] = (y_pred[i - 1] | y_pred[i + 1])

        
        
        
        
        
        
        
    t = 0
    j = 0
    for i in range(y_pred.shape[0]):
        if y_test[i].flatten().sum() > 0. and y_pred[i].flatten().sum() > 150.:
            t += numpy_dice(y_test[i], y_pred[i])
            j += 1
        elif y_test[i].flatten().sum() == 0. and y_pred[i].flatten().sum() <= 150. and y_pred[i].flatten().sum() > 0.:
            t += 1
            j += 1
        elif y_test[i].flatten().sum() == 0. and y_pred[i].flatten().sum() <= 150:
            t += 1
            j += 1
        else:
            t += 0
            j += 1
    
    print("dcs" , t/j)
    
    
    
    
    
    
    
    ##############################################################
    
    #print(model.metrics_names)

    new_y_test = new_y_test.astype('float32')
    y_pred = model.predict(new_X_test, verbose=1)
    y_pred = (y_pred > 0.1)
    y_pred = y_pred.astype(np.uint8)
    for i in range(1, y_pred.shape[0] - 1):
        y_pred[i] = cv2.morphologyEx(y_pred[i].reshape(144, 144), cv2.MORPH_CLOSE, kernel).reshape(144, 144, 1)
        y_pred[i] = cv2.morphologyEx(y_pred[i].reshape(144, 144), cv2.MORPH_OPEN, kernel).reshape(144, 144, 1)
        
        
        if np.count_nonzero(y_pred[i - 1]) > 0 and  np.count_nonzero(y_pred[i]) == 0 and np.count_nonzero(y_pred[i + 1]) > 0:
            y_pred[i] = (y_pred[i - 1] | y_pred[i + 1])
        if np.count_nonzero(y_pred[i - 1]) > 0 and  np.count_nonzero(y_pred[i]) == 0 and np.count_nonzero(y_pred[i + 1]) == 0:
            y_pred[i] = (y_pred[i - 1])
    
        
    y_pred_all = model.predict(X_test, verbose=1)
    y_pred_all = (y_pred_all > 0.1)
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    list_pix_size = []
    for i in range(num_iter):
         if y_test[i].flatten().sum() > 0:
             list_pix_size.append(np.count_nonzero(y_test[i]))
             
         if y_test[i].flatten().sum() == 0 and y_pred_all[i].flatten().sum() > 90:
             fp += 1
         if y_test[i].flatten().sum() > 0 and y_pred_all[i].flatten().sum() <= 90:
             fn += 1
         if y_test[i].flatten().sum() > 0 and y_pred_all[i].flatten().sum() > 90:
             tp += 1
         if y_test[i].flatten().sum() == 0 and y_pred_all[i].flatten().sum() <= 90:
             tn += 1
    print("MIN PIXEL", min(list_pix_size))
    print("MAX PIXEL", max(list_pix_size))
    print("MEAN PIXEL", mean(list_pix_size))
    print("All slices:", tp + tn + fp + fn)             
    print("Number of False Negatives: ", fn)
    print("Number of False Positives: ", fp)
    print("Number of True Negatives: ", tn)
    print("Number of True Positives: ", tp)
    print("Sensitivity: ", tp / (tp + fn))
    print("Specificity: ", tn / (tn + fp))
    print("Percision: ", tp / (tp + fp))
    print("Accuracy: ", (tp + tn)/(tp + tn + fp + fn))
    end_fn = 0
    end_fp = 0
    mid_fn = 0
    mid_fp = 0
    '''for i in range(1, num_iter - 2):
        if y_test[i].flatten().sum() == 0 and y_pred_all[i].flatten().sum() > 150:
            if y_pred_all[i + 1].flatten().sum() == 0 and y_pred_all[i + 2].flatten().sum() ==0:
                end_fp += 1
                print("END PIXELS", np.count_nonzero(y_pred_all[i]))
                print("END PIXELS real", np.count_nonzero(y_test[i]))
            elif y_pred_all[i - 1].flatten().sum() == 0:
                end_fp += 1
                print("END PIXELS", np.count_nonzero(y_pred_all[i]))
                print("END PIXELS real", np.count_nonzero(y_test[i]))
            else:
                mid_fp += 1
                print(i)
                print("MIDDLE PIXELS", np.count_nonzero(y_pred_all[i]))
                print("MIDDLE PIXELS real", np.count_nonzero(y_test[i]))
        if y_test[i].flatten().sum() > 0 and y_pred_all[i].flatten().sum() <= 150:
            if y_pred_all[i + 1].flatten().sum() == 0 and y_pred_all[i + 2].flatten().sum() ==0:
                end_fn += 1
                print("END PIXELS", np.count_nonzero(y_pred_all[i]))
                print("END PIXELS real", np.count_nonzero(y_test[i]))
            elif y_pred_all[i - 1].flatten().sum() == 0:
                end_fn += 1
                print("END PIXELS", np.count_nonzero(y_pred_all[i]))
                print("END PIXELS real", np.count_nonzero(y_test[i]))
            else:
                mid_fn += 1
                print(i)
                print("MIDDLE PIXELS", np.count_nonzero(y_pred_all[i]))
                print("MIDDLE PIXELS real", np.count_nonzero(y_test[i]))
    print("Number of MIDDLE MISTAKES: ", mid_fp + mid_fn)
    print("Number of BASE MISTAKES: ", end_fp + end_fn) '''   
    #Add some best and worst predictions
    img_list = [range(45, 75)]
    '''count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_best:
            break'''

    segm_pred = y_pred[img_list].reshape(-1, img_rows, img_cols)
    #print(segm_pred)
    #print(segm_pred[0].shape)
    segm_resize = np.zeros((segm_pred.shape[0], 144, 144))
    for i in range(segm_pred.shape[0]):
        segm_resize[i,:]  = resize(segm_pred[i,:],(144, 144), preserve_range=True)

    #print(segm_resize.shape)
    np.save('C:/Users/mshri/Desktop/prostate_segmentation-master/data/' + 'uhn_preds' + '.npy', segm_resize)
    img = new_X_test[img_list].reshape(-1,img_rows, img_cols)
    np.save('C:/Users/mshri/Desktop/prostate_segmentation-master/data/' + 'first_test_im' + '.npy', img)
    segm = new_y_test[img_list].reshape(-1, img_rows, img_cols).astype('float32')
    #print(segm.shape)
    fig = make_plots(img, segm, segm_pred)
    fig.savefig('C:/Users/mshri/Desktop/prostate_segmentation-master/images/best_predictions.png', bbox_inches='tight', dpi=300 )


if __name__=='__main__':
    check_predictions( )
