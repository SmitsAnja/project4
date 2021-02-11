from picamera import PiCamera
import time
from datetime import datetime
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

from PIL import Image
import cv2

#change here the path to the correct model you want to use.
model = tf.keras.models.load_model('/home/pi/project4/models/bottlesV1.tflearn')

camera = PiCamera()
camera.start_preview(fullscreen=False, window=(100,200,500,700))

for i in range(10):
    data= time.strftime("%Y-%b-%d_(%H%M%S)")
    dt = datetime.now()
    ms = dt.microsecond
    #change here the path to the correct folder you want to use for your pictures
    camera.capture('/home/pi/project4/images-timelapse/' + 'image{0}.jpg'.format(i))
    time.sleep(3)
    
    #change here the path to the correct folder where you save the pictures
    image=cv2.imread('/home/pi/project4/images-timelapse/image{0}.jpg'.format(i))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((100,100))
    p = np.expand_dims(size_image, 0)
    img = tf.cast(p, tf.float32)
    # you need to give the names of the posibility's of recognisable objects 
    # in the same order you trained the model
    # in my case it was bottles and backgrounds
    print(['bottle','background'][np.argmax(model.predict(img,steps=1))])
    
    # use this line if you want to save the pictures permanently with the timecode
    #camera.capture('/home/pi/Documents/research-project/Camera_research/timelapse/pi-timelapse/' + 'image{0}.jpg'.format(ms))

camera.stop_preview()
# if your camera is not closed or it complains about "no recources available"
# then run camera.close() seperatly
camera.close()
print("Taken 10 photo's")
