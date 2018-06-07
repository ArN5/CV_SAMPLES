
import cv2
import numpy as np
#based on this tutorial
#https://docs.opencv.org/3.4.1/d1/d5c/tutorial_py_kmeans_opencv.html

cap = cv2.VideoCapture(0)

divideFrameBy = 4

def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

cv2.createTrackbar('value 1', 'Control Panel', 4, 10, nothing)
cv2.createTrackbar('value 2', 'Control Panel', 10, 1000, nothing)
cv2.createTrackbar('value 3', 'Control Panel', 10, 1000, nothing)


ret, frame1 = cap.read()
height, width, layers = frame1.shape
# comment this line if you want the fullsize window
frame1 = cv2.resize(frame1, (int(width / divideFrameBy), int(height / divideFrameBy)))

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


while (1):

    val_1 = cv2.getTrackbarPos('value 1', 'Control Panel')
    val_2 = cv2.getTrackbarPos('value 2', 'Control Panel')
    val_3 = cv2.getTrackbarPos('value 3', 'Control Panel')


    ret, frame2 = cap.read()
    height, width, layers = frame2.shape
    # comment this line if you want the fullsize window
    frame2 = cv2.resize(frame2, (int(width / divideFrameBy), int(height / divideFrameBy)))

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)


    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2

    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    cv2.imshow('flow', hsv)


    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',bgr)

    key = cv2.waitKey(1)
    if key & 0xff == 27:
        break
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    # if the 'q' key is pressed, stop the loop
    if key == ord("Q"):
        break

cv2.destroyAllWindows()