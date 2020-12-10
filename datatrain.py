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

 # Image directory 
directory = 'traindata'


if not os.path.isdir('./' + directory):
    os.mkdir(directory)

    print('create traindata folder')




if not os.path.isfile('datatrain.csv'):

    with open('datatrain.csv', mode='w+') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(['photopath'
        ,'part_x0','part_x1','part_x2','part_x3','part_x4','part_x5','part_x6','part_x7','part_x8'
        ,'part_x9','part_x10','part_x11','part_x12','part_x13','part_x14','part_x15','part_x16'
                                
        ,'part_y0','part_y1','part_y2','part_y3','part_y4','part_y5','part_y6','part_y7','part_y8'
        ,'part_y9','part_y10','part_y11','part_y12','part_y13','part_y14','part_y15','part_y16'

        ,'score0','score1','score2','score3','score4','score5','score6','score7','score8'
        ,'score9','score10','score11','score12','score13','score14','score15','score16'

        ,'label'
    
        ])
    print('create datatrain.csv')    


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

    #check if the path not exists
    if not os.path.isdir('./'+ directory + '/' + classimage):
        #create the directory
        os.mkdir(directory + '/' + classimage)
        print('create the class directory')

    # initiaze the image path 
    image_path = directory + '/' + classimage + '/' + 'img_' + classimage + '_'


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
    
