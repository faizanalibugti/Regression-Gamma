import cv2
from grabscreen import grab_screen
import time
import numpy as np
from scipy.misc import imresize
#import keras
from tensorflow.keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform


def gamma_correction_auto(RGBimage, equalizeHist = False):
    originalFile = RGBimage.copy()
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]

    vidsize = (600, 1000)
    forLuminance = cv2.cvtColor(originalFile,cv2.COLOR_BGR2YUV)
    Y = forLuminance[:,:,0]
    totalPix = vidsize[0]* vidsize[1]
    summ = np.sum(Y[:,:])
    Yaverage = np.divide(totalPix,summ)

    correct_param = model.predict([Yaverage])

    red = red/255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    blue = blue/255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    
    output = cv2.merge((blue,green,red))
    #print(correct_param)
    return output


if __name__ == '__main__':
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # #config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras
    
    # Load Keras model
    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('regression.h5')

    while (True):
        last_time = time.time()
        screen = grab_screen(region=(0, 40, 1000, 600))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        gamma = gamma_correction_auto(screen, equalizeHist = False)

        cv2.imshow('Segmentation Gamma Correction', gamma)
        #cv2.imshow('Gamma Correction', gamma)
        #cv2.imshow('window', screen)
        print("fps: {}".format(1 / (time.time() - last_time)))

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break