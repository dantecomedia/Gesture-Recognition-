import numpy as np 
import keras
import matplotlib.pyplot as plt 
import argparse 
import os 
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate 
from keras.models import Model
import struct 
import cv2
