import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

cv2.createTrackbar('val1', 'Control Panel', 0, 255, nothing)
cv2.createTrackbar('val2', 'Control Panel', 0, 1000, nothing)
cv2.createTrackbar('val3', 'Control Panel', 0, 1000, nothing)

while (1):
    val1 = cv2.getTrackbarPos('val1', 'Control Panel')
    val2 = cv2.getTrackbarPos('val2', 'Control Panel')
    val3 = cv2.getTrackbarPos('val3', 'Control Panel')

    # Take each frame
    ret, frame = cap.read()


    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    edges = cv2.dilate(edges, None)

    cv2.imshow('edges', edges)
    minLineLength = 100+val1
    maxLineGap = 10+val2
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    GREEN = (0, 255, 0)
    try:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(img, (x1, y1), (x2, y2), GREEN , 2)
    except:
        print("no lines")

    cv2.imshow('houghlines', img)

    #cv2.imshow('edges', magnitude_spectrum)

    keyPressed = cv2.waitKey(1) & 0xFF

    if keyPressed == ord('q'):
        break

cv2.destroyAllWindows()
