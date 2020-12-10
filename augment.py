import cv2
import  random
import numpy as np
from tf_pose import common
import os

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img

def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 0.5)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img


def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


def get_files(directory):
    results = []
    for file_name in os.listdir(directory):
        tmp, ext = os.path.splitext(file_name)
        if ext == '.jpg':
            results.append(file_name)
    return results

CLASS = 'knee_touch'

INPUT_DIRECTORY = 'traindata/' + CLASS
OUTPUT_DIRECTORY = 'augment_image/' + CLASS

ANGLE = 16
if __name__ == '__main__':

    for f in get_files(INPUT_DIRECTORY):

        print('Process image: %s...' % f)

        img = common.read_imgfile(os.path.join(INPUT_DIRECTORY, f), None, None)


        output_image = rotation(img,ANGLE)
        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, 'rota_' + f), output_image)
        print('save file',os.path.join(OUTPUT_DIRECTORY, 'rota' + f))

            
        output_image = horizontal_flip(img,1)
        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, 'flip_' + f), output_image)
        print('save file',os.path.join(OUTPUT_DIRECTORY, 'flip' + f))






