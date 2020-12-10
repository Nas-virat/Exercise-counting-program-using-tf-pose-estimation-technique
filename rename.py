import os, sys

import argparse


def get_files(directory):
    results = []
    for file_name in os.listdir(directory):
        tmp, ext = os.path.splitext(file_name)
        if ext == '.jpg':
            results.append(file_name)
    return results



class_name = 'squats_down'

INPUT_DIRECTORY = 'traindata/' + class_name



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='rename image')
    parser.add_argument('--start', type=int, default=0)

    args = parser.parse_args()

    start = args.start

    for f in get_files(INPUT_DIRECTORY):
        new_file_name = 'img_'+ class_name + '_' + str(start) + '.jpg'
        os.rename(os.path.join(INPUT_DIRECTORY , f),os.path.join(INPUT_DIRECTORY ,new_file_name))
        print("File renamed!",new_file_name)
        start += 1