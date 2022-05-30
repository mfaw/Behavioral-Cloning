import argparse
import base64
from datetime import datetime
import os
import shutil
import math
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import matplotlib.pyplot as plt

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# def random_brightness(image):
#     image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
#     random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
#     image1[:,:,2] = image1[:,:,2]*random_bright
#     image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
#     return image1
# def cropImage(image):
#     imageShape = image.shape
#     startColumn = 20 
#     endColumn = imageShape[1] - 20
#     startRow = 60
#     endRow = imageShape[0] - 24
    
#     x = np.random.randint(-20 , 20)
#     y = np.random.randint(-10 , 10)
    
#     img = image[ startRow+y:endRow+y , : , :]
#     img = cv2.resize(img , (200 , 66) , cv2.INTER_AREA)
# #     angle += -x/(40)/3.0
    
#     return img 

def random_shear(image,shear_range = 100):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    
    return image
    

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

MAX_SPEED = 10.5
MIN_SPEED = 10.3
speed_limit = MAX_SPEED

controller = SimplePIController(0.08, 0.004)
set_speed = 9
controller.set_desired(set_speed)

def resize( image, new_dim):
        return cv2.resize(image , new_dim)
def crop( image, top_percent, bottom_percent):

        top = int(np.ceil(image.shape[0] * top_percent))
        bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

        return image[top:bottom, :]
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        global MAX_SPEED 
        global MIN_SPEED 
        global speed_limit
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        # image_array = random_brightness(image_array)
        image_array = crop(image_array,0.38 , 0.137)
        image_array = cv2.resize(image_array , (64 , 64))
        image_array = cv2.cvtColor(image_array,cv2.COLOR_RGB2HSV)
        # image_array = cv2.cvtColor(image_array,cv2.COLOR_RGB2HSV)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        if float(speed) > speed_limit:

            speed_limit = MIN_SPEED
        else:
            speed_limit = MAX_SPEED
        
        throttle_2 = controller.update(float(speed))
        throttle_1 = 1.0 - steering_angle**2 - (float(speed)/speed_limit)
        # throttle = 0.8 - steering_angle**2 if float(speed) < 10 else .2 - steering_angle**2
        # throttle = (26-np.float32(speed))*0.5
        if(throttle_2<throttle_1):
            throttle = throttle_2
            print(steering_angle, throttle , "T2")
        else:
            throttle = throttle_1
            print(steering_angle, throttle , "T1")
        if(throttle == 0):
            throttle = 0.01
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
