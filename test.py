'''
    this program create the data for train
    a a classofier model in term of csv

    create by Napas vinitnantharat 
        update 15 Nov 2020
'''

import argparse #for line arguement
import logging #loggine to terminal
import time #time control module

import cv2 #module for image processing
import numpy as np #array module
import csv #to write the data to csv file

import tensorflow as tf #machine learning module

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import os #module operation system


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score




part_x = [0]*17
part_y = [0]*17
score_point = [0]*17

df = pd.read_csv('datatrain.csv')

df_drop = df.drop(columns = ['photopath','label'])

X = df_drop
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=300)

clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

fps_time = 0

cam = cv2.VideoCapture(0)

out = ''

MODEL = 'mobilenet_thin'

e = TfPoseEstimator(get_graph_path(MODEL), target_size=(432, 368))

while True:

    ret_val, image = cam.read()

    humans = e.inference(image, resize_to_default=False, upsample_size=4)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    try:
        #for each person collect all the joint of that person
        for j in range(0,17):
            part_x[j] = humans[0].body_parts[j].x*image.shape[1]
            part_y[j] = humans[0].body_parts[j].y*image.shape[0] 
            score_point[j] = humans[0].body_parts[j].score
            total_score =  humans[0].score
    except:
         pass

    out = clf.predict([part_x + part_y + score_point])                                 

    print(out)
    print([part_x + part_y + score_point])

    cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    fps_time = time.time()


    cv2.imshow('tf-pose-estimation result', image)
        
    if cv2.waitKey(1) == 27:
        break



