import os, sys

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


import csv

classimage = 'squats_down'

INPUT_DIRECTORY = 'traindata/' + classimage

MODEL = 'mobilenet_thin'



def get_files(directory):
    results = []
    for file_name in os.listdir(directory):
        tmp, ext = os.path.splitext(file_name)
        if ext == '.jpg':
            results.append(file_name)
    return results



part_x = [0]*17
part_y = [0]*17
score_point = [0]*17


if __name__ == '__main__':
    #640x480
    e = TfPoseEstimator(get_graph_path(MODEL), target_size=(640, 480))

    for f in get_files(INPUT_DIRECTORY):
        print('Process image: %s...' % f)
        image = common.read_imgfile(os.path.join(INPUT_DIRECTORY, f), None, None)
        humans = e.inference(image, resize_to_default=False, upsample_size=4)

        try:
            #for each person collect all the joint of that person
            for j in range(0,17):
                part_x[j] = humans[0].body_parts[j].x
                part_y[j] = humans[0].body_parts[j].y
                score_point[j] = humans[0].body_parts[j].score
                total_score =  humans[0].score
        except:
            pass

        with open('datatrainall.csv', mode='a') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow([f] + part_x + part_y + score_point +[classimage])

            print('upload csv', f)
            #save image to fold
        

    
        