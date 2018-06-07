
import cv2
import numpy as np
#based on this tutorial
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html

cap = cv2.VideoCapture(0)


def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

cv2.createTrackbar('maxCorners', 'Control Panel', 25, 255, nothing)
cv2.createTrackbar('qualityLevel', 'Control Panel', 10, 1000, nothing)
cv2.createTrackbar('minDistance', 'Control Panel', 10, 1000, nothing)


while (1):

    maxCorners = cv2.getTrackbarPos('maxCorners', 'Control Panel')
    qualityLevel = cv2.getTrackbarPos('qualityLevel', 'Control Panel')
    minDistance = cv2.getTrackbarPos('minDistance', 'Control Panel')

    _, frame = cap.read()

    height, width, layers = frame.shape

    # comment this line if you want the fullsize window
    frame = cv2.resize(frame, (int(width / 2), int(height / 2)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #added
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, 1/(1+qualityLevel), minDistance)
    #(image=,maxCorners=,qualityLevel=,minDistance=,corners=,mask=,blockSize=,useHarrisDetector=,k=)
    #maxCorners – Maximum number of corners to return. If there are more corners than are found, the strongest of
    #them is returned.

    #qualityLevel – Parameter characterizing the minimal accepted quality of image corners. The parameter value is
    #multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() )
    # or the Harris function response (see cornerHarris() ). The corners with the quality measure less than
    #the product are rejected. For example, if the best corner has the quality measure = 1500, and the
    #qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.

    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 2, 255, -1)


    cv2.imshow('Control Panel', frame)

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