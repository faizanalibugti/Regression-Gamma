import cv2
from grabscreen import grab_screen
import time
import numpy as np
from scipy.misc import imresize
from getkeys import key_check
import os

def gamma_correction_auto(RGBimage, equalizeHist = False):
    originalFile = RGBimage.copy()
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]

    vidsize = (1000, 600)
    forLuminance = cv2.cvtColor(originalFile,cv2.COLOR_BGR2YUV)
    Y = forLuminance[:,:,0]
    totalPix = vidsize[0]* vidsize[1]
    summ = np.sum(Y[:,:])
    Yaverage = np.divide(totalPix, summ)
    print("Y: {}".format(Yaverage))

    epsilon = 1.19209e-007
    correct_param = np.divide(-0.3,np.log10([Yaverage + epsilon]))
    correct_param = 0.7 - correct_param 
    print("Enhance: {}".format(correct_param[0]))

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
    
    return output, Yaverage, correct_param[0]


if __name__ == '__main__':
    file_name = 'training_data.npy'

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):
        if not paused:
            last_time = time.time()
            screen = grab_screen(region=(0, 40, 640, 480))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            gamma, Yaverage, correct_param = gamma_correction_auto(screen, equalizeHist = False)
            
            training_data.append([Yaverage,correct_param])

            if len(training_data) % 100 == 0:
                print(len(training_data))
                np.save(file_name,training_data)

            cv2.imshow('Gamma Correction', gamma)
            print("fps: {}".format(1 / (time.time() - last_time)))

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
        
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

