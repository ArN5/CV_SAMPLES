import numpy as np
import cv2

cap = cv2.VideoCapture('Semi_Final_Run.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    #convert to grayscale to show that
    # you can do operations on the video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()