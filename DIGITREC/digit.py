import cv2
import numpy as np
from PIL import ImageGrab #for screen capture
import imutils #resizing

def imageGrabVideo():

    #bbox specifies specific region (bbox= x,y,width,height)
    img = ImageGrab.grab(bbox=(100,190,500,500))
    
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

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

#model = cv2.KNearest()
model = cv2.ml.KNearest_create()

#model.train(samples,responses)
model.train(samples,cv2.ml.ROW_SAMPLE,responses)

############################# testing part  #########################

while(1):

    #im = cv2.imread('pi.png')
    #grab screen cap video
    im = imageGrabVideo()
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    #find the contours of the numbers
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)



    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            #draw rectangles around the centers of countours
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>28:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)

                
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
    


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    cv2.imshow('im',im)
    cv2.imshow('out',out)

cv2.waitKey(0)