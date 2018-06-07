
import cv2

cap = cv2.VideoCapture(0)

divideFrameBy = 2

backGroundSubKNN = cv2.createBackgroundSubtractorKNN()


while(1):

    ret, frame = cap.read()
    height, width, layers = frame.shape
    # comment this line if you want the fullsize window
    frame = cv2.resize(frame, (int(width / divideFrameBy), int(height / divideFrameBy)))

    fbgsKNN = backGroundSubKNN.apply(frame)

    cv2.imshow('fgmask', fbgsKNN)
    cv2.imshow('fg2mask', fbgsMOG)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


