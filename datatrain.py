'''
    this program create the data for train
    a a classofier model in term of csv

    create by Napas vinitnantharat 
        update 10 DEC 2020
'''

import argparse #for line arguement
import logging #loggine to terminal
import time #time control module

import cv2 #module for image processing
import numpy as np #array module
import csv #to write the data to csv file

import os #module operation system
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


#count the number of frame that have past
loop = 0
#index of the image class
index = 0

MODEL = 'mobilenet_thin'

fps_time = 0


DIRECTORY  = 'traindata'

cam = cv2.VideoCapture(0)


if __name__ == '__main__':
    '''
    use the line arguement 
    '''
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    #napas add arguement classimage
    parser.add_argument('--numimage',type=int,default=150,
                        help=' number of image to record')

    parser.add_argument('--classimage',type=str,default='action1',
                        help='class image that we want to classifier')

    #set up the variable from line arguement
    args = parser.parse_args()

    classimage = args.classimage
    num_image =  args.numimage

    
    # initiaze the image path 
    image_path = DIRECTORY + '/' + classimage + '/' + 'img_' + classimage + '_'


    # find the last index  of the png
    while os.path.isfile(image_path + str(index) +'.jpg') :
        index += 1

    e = TfPoseEstimator(get_graph_path(MODEL), target_size=(640, 480))
    #main loop 
    while loop < num_image:
        #read the image from web_cam
        time.sleep(1)

        ret_val, image = cam.read()
        #capture the webcam image to the directory class 
        status = cv2.imwrite( image_path + str(index) +'.jpg', image)
        if status:
            print('sucess record to ',image_path + str(index) +'.jpg')
        else:
            print('not record to', image_path + str(index) +'.jpg')
        
        humans = e.inference(image, resize_to_default=False, upsample_size=4)
        output_image = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)
        cv2.imshow('tf-pose-estimation result', output_image)
        if cv2.waitKey(1) == 27:
            break
        loop += 1 
        index += 1

    cv2.destroyAllWindows()
    
