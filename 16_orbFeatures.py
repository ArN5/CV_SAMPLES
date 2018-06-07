
import cv2
import numpy as np
#based on this tutorial
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html

cap = cv2.VideoCapture(0)

# Initiate STAR detector
orb = cv2.ORB_create()

def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

cv2.createTrackbar('value 1', 'Control Panel', 25, 255, nothing)
cv2.createTrackbar('value 2', 'Control Panel', 10, 1000, nothing)
cv2.createTrackbar('value 3', 'Control Panel', 10, 1000, nothing)


while (1):

    val_1 = cv2.getTrackbarPos('value 1', 'Control Panel')
    val_2 = cv2.getTrackbarPos('value 2', 'Control Panel')
    val_3 = cv2.getTrackbarPos('value 3', 'Control Panel')

    _, frame = cap.read()

    height, width, layers = frame.shape

    # comment this line if you want the fullsize window
    frame = cv2.resize(frame, (int(width / 2), int(height / 2)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the keypoints with ORB
    kp = orb.detect(frame, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(frame, kp)
    #draw the keypoints
    kpF = cv2.drawKeypoints(frame, kp[:val_1],None ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #kpF = cv2.drawKeypoints(frame, kp[:val_1],None, color=(0,0,255) ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('kp', kpF)



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