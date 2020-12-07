'''
    this program create the data for train
    a a classofier model in term of csv

    create by Napas vinitnantharat 
        update 27 Nov 2020
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
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


from joblib import dump
# add for gpu
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0],True)
'''


part_x = [0]*17
part_y = [0]*17
score_point = [0]*17

df = pd.read_csv('datatrainall.csv')
df_drop = df.drop(columns = ['photopath','label'])


X = df_drop
y = df[['photopath', 'label']]

print(df['label'].value_counts())
print('\n\n')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=300)

y_train_photopath = y_train['photopath']
y_train = y_train['label']

y_test_photopath = y_test['photopath']
y_test = y_test['label']

clf = MLPClassifier(hidden_layer_sizes=(64, 64,), max_iter=1000, solver='adam').fit(X_train, y_train)

joblib_file = 'action_predict.pkl'

dump(clf,joblib_file)


y_pred = clf.predict(X_train)

for i in range(len(y_pred)):
    if y_pred[i] != y_train.values[i]:
        print(y_train_photopath.values[i], 'predict:', y_pred[i], 'actual:', y_train.values[i])


print('\n\n')
y_pred = clf.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i] != y_test.values[i]:
        print(y_test_photopath.values[i], 'predict:', y_pred[i], 'actual:', y_test.values[i])