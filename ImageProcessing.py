from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image
import numpy as np
import os
from scipy import ndimage 
from scipy.misc import imread, imsave, imresize

# Make sure to define paths when calling to methods using folders with images.

#%%
def createTrainingImages(path_GT, path_LR):    
    if len(os.listdir(path_LR)) == 0:   
        list_import = os.listdir(path_GT)
    
        for file in list_import:
            im = imread(path_GT + '//' + file)
            im_rows = im.shape[0]
            im_cols = im.shape[1]
            
            im = imresize(im, (int(im_rows/2), int(im_cols/2)), interp='bilinear')
            im = imresize(im, (im_rows, im_cols), interp='bicubic')
            im = ndimage.gaussian_filter(im, 2)
            imsave(path_LR + "//" + "%s" % file, im)           

#%%
def getImageMatrix(path):
    list = os.listdir(path)
    
    matrix = np.array([np.array(Image.open(path + '//' + file))
                    for file in list], 'f')
    
    matrix = matrix.reshape(matrix.shape[0], 1, matrix.shape[1], matrix.shape[2])
    
    return matrix

#%%    
def getImages(path):
    list = os.listdir(path)
    
    images = np.array([np.array(Image.open(path + '//' + file))
                    for file in list], 'f')
    
    return images

#%%
def getVariables(LR_matrix, GT_matrix):
    data, label = shuffle(LR_matrix, GT_matrix, random_state=4)
    train_data = [data, label]
    
    (X, Y) = (train_data[0], train_data[1])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    Y_train /= 255
    Y_test /= 255
    
    return X_train, X_test, Y_train, Y_test

#%%  
def getJointVariables(HP_matrix, HP_GT_matrix, proton_matrix):
    data_HP, data_proton, data_HP_GT = shuffle(HP_matrix, proton_matrix, HP_GT_matrix, random_state=4)
    
    X_hp_train, X_hp_test, X_proton_train, X_proton_test, Y_train, Y_test = train_test_split(data_HP, data_proton, data_HP_GT, test_size=0.2, random_state=5)
    X_hp_train = X_hp_train.astype('float32')
    X_hp_test = X_hp_test.astype('float32')
    X_hp_train /= 255
    X_hp_test /= 255
    
    X_proton_train = X_proton_train.astype('float32')
    X_proton_test = X_proton_test.astype('float32')
    X_proton_train /= 255
    X_proton_test /= 255
    
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    Y_train /= 255
    Y_test /= 255
    
    return X_hp_train, X_hp_test, X_proton_train, X_proton_test, Y_train, Y_test

#%%
def getResizedImageMatrix(path, size):
    list = os.listdir(path)
    matrix = []
    
    for file in list:
        im = imread(path + '//' + file)
        im = imresize(im, (size, size), interp="bicubic")
        matrix.append(np.array(im))

    matrix = np.array(matrix.reshape(matrix.shape[0], 1, matrix.shape[1], matrix.shape[2]))    
    
    return matrix

#%%
def saveImages(path_export, decoded_images):   
    j = len(decoded_images)
    for i in range(j):
        predicted = decoded_images[i].reshape(decoded_images.shape[2], decoded_images.shape[3])        
        imsave(path_export + "//" + "predicted" + "%s.png" % str(i), predicted)
        
#%%
def getTrainingImages(path_GT, path_LR):        
    createTrainingImages(path_GT, path_LR)
        
    LR_matrix = getImageMatrix(path_LR)
    GT_matrix = getImageMatrix(path_GT)
    X_train, X_test, Y_train, Y_test = getVariables(LR_matrix, GT_matrix)   
    
    return X_train, X_test, Y_train, Y_test

#%%
def getJointTrainingImages(path_HP, path_proton, path_HP_GT, size):   
    createTrainingImages(path_HP_GT, path_HP)
        
    HP_matrix = getResizedImageMatrix(path_HP, size)
    proton_matrix = getResizedImageMatrix(path_proton, size)
    HP_GT_matrix = getResizedImageMatrix(path_HP_GT, size)
    X_hp_train, X_hp_test, X_proton_train, X_proton_test, Y_train, Y_test = getJointVariables(HP_matrix, HP_GT_matrix, proton_matrix)   
    
    return X_hp_train, X_hp_test, X_proton_train, X_proton_test, Y_train, Y_test 