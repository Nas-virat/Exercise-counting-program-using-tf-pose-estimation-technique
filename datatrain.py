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

# intial all point 
part_x = [0]*17
part_y = [0]*17
total_score = 0
score_point = [0]*17

#the number of the data row we want to collect 
row_data = 30
#count the number of frame that have past
loop = 0
#index of the image class
index = 0

bodypartid =[0]*17

fps_time = 0

 # Image directory 
directory = 'traindata'


if not os.path.isdir('./' + directory):
    os.mkdir(directory)

    print('create traindata folder')

keys_point  ={
    "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
"Background": 15
}



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


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    '''
    use the line arguement 
    '''
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    #napas add arguement classimage
    parser.add_argument('--classimage',type=str,default='action1',
                        help='class image that we want to classifier')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')


    #set up the variable from line arguement
    args = parser.parse_args()

    classimage = args.classimage

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

    #width and height
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()




    #main loop 
    while loop < row_data:
        #read the image from web_cam
        time.sleep(1)

        ret_val, image = cam.read()
        #capture the webcam image to the directory class 
        status = cv2.imwrite( image_path + str(index) +'.jpg', image)
        if status:
            print('sucess record to ',image_path + str(index) +'.jpg')
        else:
            print('not record to', image_path + str(index) +'.jpg')
        
        #pose estimator
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

       
        
        try:
            #for each person collect all the joint of that person
            for j in range(0,17):
                part_x[j] = humans[0].body_parts[j].x*image.shape[1]
                part_y[j] = humans[0].body_parts[j].y*image.shape[0] 
                score_point[j] = humans[0].body_parts[j].score
                bodypartid[j] = humans[0].body_parts[j].part_idx
                total_score =  humans[0].score
        except:
            pass
        print('append data',index,'to csv')

        #append the data to csv 
        with open('datatrain.csv', mode='a') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow([image_path + str(index) +'.jpg'] + part_x + part_y + score_point +[classimage])

        # print the fps on left top
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        loop += 1 
        index += 1

    cv2.destroyAllWindows()
    
