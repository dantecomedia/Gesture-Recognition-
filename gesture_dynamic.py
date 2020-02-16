#loading the basic dependencies
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image
import sklearn


model=load_model('weights_gesture.h5')

def prediction(pred):
    return (chr(pred + 65))

def keras_predict(model, image):
    data= np.asarray(image, dtype="int32")
    pred_probab=model.predict(data)[0]
    pred_class=list(pred_probab)

def main():
    while True:
        cam_capture=cv2.VideoCapture(0)
        _,image_frame=cam_capture.read(0)
        im2= crop_image(image_frame, 300, 300, 300, 300)
        image_grayscale=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
        image_grayscale_blurred=cv2.GaussianBlur(image_grayscale,(15,15),0)
        im3=cv2.resize(image_grayscale_blurred,(28,28), interpolation=cv2.INTER_AREA)
        im4=np.resize(im3,(28,28,1))
        im5=np.expand_dims(im4, axis=0)
