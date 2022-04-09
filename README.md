# exercise counting program using tf-pose-estimation technique 
this is a program counting exercise using pose estimation technique. the exercise including squats touch feet

## Install
Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd CPE101
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess

$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk
- slidingwindow

### run the program
```
$ python3 test.py

```
### Project Document
file :final_report.pdf

### Result of the project
https://drive.google.com/file/d/18bmnzpSfJotYEUWdAwqH1hWHL9P5sBRp/view?usp=sharing

### credit 
https://github.com/ildoonet/tf-pose-estimation
