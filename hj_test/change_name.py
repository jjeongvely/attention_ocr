import os, sys
from PIL import Image
import cv2
import pandas as pd

#image directory path
IMG_DIR = '/home/qisens/tensorflow/models/research/attention_ocr/UK/hj_test/platesMania_test_150by150'

i=0
for path in os.listdir(IMG_DIR):
    img = cv2.imread(IMG_DIR +'/' + path)

    cv2.imwrite(IMG_DIR + '/' + 'number_plates_%02d'%i+'.png', img)
    i+=int(1)
