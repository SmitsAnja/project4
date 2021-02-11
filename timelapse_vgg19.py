from picamera import PiCamera
import time
from datetime import datetime

from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16

camera = PiCamera()
model = VGG16()

#camera.close()
camera.start_preview(fullscreen=False, window=(100,200,500,700))

#def model_vgg19:
    

for i in range(10):
    data= time.strftime("%Y-%b-%d_(%H%M%S)")
    dt = datetime.now()
    ms = dt.microsecond
    camera.capture('/home/pi/project4/images-timelapse/' + 'image{0}.jpg'.format(i))
    time.sleep(0.2)
    
    
    # load an image from file
    image = load_img('/home/pi/project4/images-timelapse/' + 'image{0}.jpg'.format(i), target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))
    time.sleep(1)

camera.stop_preview()
camera.close()
print("Taken 10 photo's")
