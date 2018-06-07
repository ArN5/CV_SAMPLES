import sys

import numpy as np
import cv2
#use pillow
from PIL import ImageGrab #for screen capture
import imutils #resizing

def imageGrabVideo():

    #bbox specifies specific region (bbox= x,y,width,height)
    img = ImageGrab.grab(bbox=(100,190,600,600))
    
    #this is very important or else the colors will
    #not match and you will have a hard time    
    img = img.convert('RGB')

    #convert the image into a numpy array
    img_np = np.array(img)

    #resize the image array
    frame = imutils.resize(img_np, width=500)
    
    #important for color matching turns RGB into BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    return frame


#im = cv2.imread('pitrain.png')
im = imageGrabVideo()
im3 = im.copy()
cv2.imshow('norm',im)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#_,
cv2.imshow('thresh',thresh)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>70:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>10:#28
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,255),2)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print ("training complete")

np.savetxt('generalsamples.data',samples)
#np.savetxt('generalsamples.txt',samples)
np.savetxt('generalresponses.data',responses)
#np.savetxt('generalresponses.txt',responses)