import os, sys

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

INPUT_DIRECTORY = 'traindata/set'
OUTPUT_DIRECTORY = 'output_images/set'

MODEL = 'mobilenet_thin'



def get_files(directory):
    results = []
    for file_name in os.listdir(directory):
        tmp, ext = os.path.splitext(file_name)
        if ext == '.jpg':
            results.append(file_name)
    return results




if __name__ == '__main__':

    e = TfPoseEstimator(get_graph_path(MODEL), target_size=(432, 368))
    for f in get_files(INPUT_DIRECTORY):
        print('Process image: %s...' % f)
        image = common.read_imgfile(os.path.join(INPUT_DIRECTORY, f), None, None)
        humans = e.inference(image, resize_to_default=False, upsample_size=4)
        output_image = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)
        
        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, f), output_image)

    print('Done.')
