# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:53:23 2019

@author: utente
"""

from FitMonitor import *
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from utils import load_and_resize,read_categories_dict
import pickle
import os
import shutil


def learning(X_train,y_train,X_test,y_test,nb_classes=10):
    nb_epoch = 140
    batch_size = 32
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
               
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation='softmax'))
    
    
    # Compile model with SGD (Stochastic Gradient Descent)
    lrate = 0.01
    decay = lrate/nb_epoch
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
    model.load_weights("weights.best.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    
    print('Augmented Data Training.')
    
    imdgen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False,  # randomly flip images
    )
    
    imdgen.fit(X_train)
    
    # fit the model on the batches generated by datagen.flow()
    dgen = imdgen.flow(X_train, Y_train, batch_size=batch_size)
    filepath="weights.best.hdf5"
    check=ModelCheckpoint(filepath,monitor='val_acc',verbose=2,save_best_only=True, mode='max')
    h = model.fit_generator(
        dgen,
        samples_per_epoch = X_train.shape[0],
        nb_epoch = nb_epoch,
        validation_data = (X_test, Y_test),
        verbose = 0,
        callbacks = [FitMonitor(thresh=0.03,minacc=0.99),check]
    )
    
    print('Saving model to "model.h5"')
    model.save("model.h5")
   
    loss, acc = model.evaluate(X_train, Y_train, verbose=0)
    print("Training: accuracy   = %.6f loss = %.6f" % (acc, loss))
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print("Validation: accuracy = %.6f loss = %.6f" % (acc, loss))
    print('Saving history dict to pickle file: hist.p')

    with open('hist.p', 'wb') as f:
        pickle.dump(h.history, f)
        
        
def classifier(dirpath):
    if os.path.exists(dirpath):
        if os.path.isdir(dirpath):
            
           try:
               x_test=load_and_resize(dirpath)
               model=load_model('model.h5')
               x_test=x_test.astype('float32')
               x_test /= 255
               prediction=model.predict_classes(x_test)
           except ValueError:
               raise ValueError
           
        else: print('Error - Not a directory')
    else: print('Error - Path not exist')
    return prediction


def summary(dirpath):
    categories=read_categories_dict('categories10.txt')
    pred=classifier(dirpath)
    
    for image_file,p in zip(os.listdir(dirpath),pred):
        print('%s: %s' % (image_file,categories.get(p)[0]))
        if os.path.exists('Images/' + categories.get(p)[0]):
            dest='Images/' + categories.get(p)[0]
            shutil.move(dirpath+'/'+image_file,dest)
            
        else:
            dest='Images/' + categories.get(p)[0]
            os.makedirs(dest)
            shutil.move(dirpath+'/'+image_file,dest)

            
            
            
  
        
def testing(X_test,y_test):
  model=load_model('model.h5')
  y_pred=model.predict_classes(X_test)
  true_preds = [(x,y) for (x,y,p) in zip(X_test, y_test, y_pred) if y == p]
  false_preds = [(x,y,p) for (x,y,p) in zip(X_test, y_test, y_pred) if y != p]
  print("Number of true predictions: ", len(true_preds))
  print("Number of false predictions:", len(false_preds))


'''(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
testing(X_test,Y_test)
summary('Test_images')'''