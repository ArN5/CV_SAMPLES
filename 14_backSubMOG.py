
import cv2
import numpy as np
#based on this tutorial
#https://docs.opencv.org/3.4.1/d1/d5c/tutorial_py_kmeans_opencv.html

cap = cv2.VideoCapture(0)

divideFrameBy = 4

backGroundSubMOG = cv2.createBackgroundSubtractorMOG2()

while(1):

    ret, frame = cap.read()
    height, width, layers = frame.shape
    # comment this line if you want the fullsize window
    frame = cv2.resize(frame, (int(width / divideFrameBy), int(height / divideFrameBy)))

    fbgsMOG = backGroundSubMOG.apply(frame)

    colorCutout =  cv2.bitwise_and(frame, frame, mask=fbgsMOG)
    cv2.imshow('colorCutout', colorCutout)

    cv2.imshow('fbgsMOG',fbgsMOG)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()