import cv2
import numpy as np
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Semi_Final_Run.mp4')

def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('window')

# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('threshold-1', 'window', 3, 20, nothing)
cv2.createTrackbar('threshold-2', 'window', 4, 200, nothing)

frameNumber = 0

while (1):
    # Take each frame
    frameNumber = frameNumber + 1
    cap.set(1, frameNumber)
    ret, frame = cap.read()

    #frame = cv2.medianBlur(frame, 5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.equalizeHist(frame)

    #frame = cv2.bilateralFilter(frame, 9, 75, 75)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    # get info from the trackbars
    th1 = cv2.getTrackbarPos('threshold-1', 'window')

    th2 = cv2.getTrackbarPos('threshold-2', 'window')

    th3 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2*th1+3, 0.1*th2)
    th3 = cv2.medianBlur(th3, 5)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cv2.imshow('window', th3)

    keyPressed = cv2.waitKey(1) & 0xFF

    if keyPressed == ord('q'):
        break
    if keyPressed == ord('o'):
        frameNumber = frameNumber - 190
    if keyPressed == ord('p'):
        frameNumber = frameNumber + 190

cv2.destroyAllWindows()
