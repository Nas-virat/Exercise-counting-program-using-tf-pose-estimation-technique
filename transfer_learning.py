''''
    

    Created by Napas Vinitnantharat FANG

'''
import time #time control module

import cv2 #module for image processing
import numpy as np #array module
import csv #to write the data to csv file

import tensorflow as tf #machine learning module
from tensorflow import keras
from tensorflow.keras import layers

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import os #module operation system
import random

from keras.models import Sequential
from keras.layers import Dense


import matplotlib.pyplot as plt
import numpy as np


DATA = 'traindata'

classes = ['squats_down','set','knee_touch']

training_Data = []

X = []
y = []

def create_training_data():
    for category in classes:
        path = os.path.join(DATA,category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(640,480))
                training_Data.append([new_array,class_num])
            except  Exception as e:
                pass



if __name__ == '__main__':

    create_training_data()

    random.shuffle(training_Data)

    for features , label in training_Data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1,640,480,3)

    X = X/255.0

    Y = np.array(y)
    
    model = tf.keras.applications.mobilenet.MobileNet()

    
    #transfer learning 
    base_input = model.layers[0].input
    base_output = model.layers[-4].output

    Flat_layer = layers.Flatten()(base_output)
    final_output = layers.Dense(3)(Flat_layer)
    final_ouput  = layers.Activation('sigmoid')(final_output) 

    new_model = keras.Model(inputs = base_input,outputs = final_output)

    new_model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
    new_model.fit(X,y,epochs =20 ,validation_spilt = 0.2)

    new_model.save('model.h5')

    