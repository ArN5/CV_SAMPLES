import cv2

cap = cv2.VideoCapture(0)

divideFrameBy = 4
# save the image frame
ret, frame = cap.read()
#find the size
height, width, channel = frame.shape

print("width:", width, " height:", height, "channels:", channel )
print("Frames per Second: ",cap.get(cv2.CAP_PROP_FPS))


while (1):

    # save the image frame
    ret, frame = cap.read()

    # resize image
    frame = cv2.resize(frame, (int(width / divideFrameBy), int(height / divideFrameBy)))

    # show the image frame
    cv2.imshow('frame', frame)

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
