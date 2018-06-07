import cv2

cap = cv2.VideoCapture(0)


def findBiggestContour(mask):
    contoursArray = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    # only proceed if at least one contour was found
    if len(contoursArray) > 0:
        biggestCountour = max(contoursArray, key=cv2.contourArea)
        return biggestCountour

while (1):
    # save the image frame
    ret1, frame_1 = cap.read()
    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

    ret2, frame_2 = cap.read()
    frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)


    frameDelta = cv2.absdiff(frame_1, frame_2)

    frame  = frame_1

    #frameDelta = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)



    ret, thresh = cv2.threshold(frameDelta,40,255,cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    cv2.imshow('frame', frameDelta)


    # show the image frame and
    # wait for 1 milliseconds
    # to check if user click "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if user clicks "q" then break out of the loop

# make sure to release the camera
cap.release()
# close all windows
cv2.destroyAllWindows()
