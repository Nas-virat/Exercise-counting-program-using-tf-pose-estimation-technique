import os, sys
import shutil
from tf_pose import common
import cv2
import numpy as np

import time

CLASS = 'lunges_down'


INPUT_DIRECTORY = 'output_images/' + CLASS


def get_files(directory):
    results = []
    for file_name in os.listdir(directory):
        tmp, ext = os.path.splitext(file_name)
        if ext == '.jpg':
            results.append(file_name)
    return results


if __name__ == '__main__':


    for f in get_files(INPUT_DIRECTORY):
        image = common.read_imgfile(os.path.join(INPUT_DIRECTORY, f), None, None)
        cv2.imshow('select',image)
        print('b: backimage\n')
        print('o: ok image\n')
        select = str(input("please enter:"))

        if(select == 'b'):
            shutil.move(os.path.join(INPUT_DIRECTORY,f), 'bad_image/' + CLASS)


            
