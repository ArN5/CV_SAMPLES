import numpy as np
import cv2



def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

cv2.createTrackbar('threshold-1', 'Control Panel', 5, 1000, nothing)

cap = cv2.VideoCapture(0)

while(1):
    th1 = cv2.getTrackbarPos('threshold-1', 'Control Panel')

    _, frame = cap.read()
    height, width, layers = frame.shape
    frame = cv2.resize(frame, (int(width / 2), int(height / 2)))

    # pyr = cv2.pyrMeanShiftFiltering(frame, 21, 51)
    # cv2.imshow("pyr", pyr)



    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # sure qqbackground area
    #sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    cv2.imshow("dist", dist_transform)

    ret, sure_fg = cv2.threshold(dist_transform, 1/(th1+1) * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)


    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1


    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(frame, markers)


    frame[markers == -1] = [0, 0, 255]#make red markers

    cv2.imshow("frame", frame)

    segm =  cv2.bitwise_and(frame, frame, mask=thresh)
    cv2.imshow('segm', segm)


    # wait for keyboard to be pressed store the value as key
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break



cv2.destroyAllWindows()