import cv2

cap = cv2.VideoCapture(0)

while (1):
    # save the image frame
    ret, frame = cap.read()

    # show the image frame
    cv2.imshow('frame', frame)
    #/////////////////////////////

    # create a new frame in
    # "Hue Saturation Value" or HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)


    #create a boundary for the dark red
    redLower = (160, 100, 100)
    redUpper = (180, 255, 255)
    #( Hue 0-180, Saturation 0-255, Value 0-255 )

    # create a new filtered frame that only captures dark red objects
    filteredFrame = cv2.inRange(hsv, redLower, redUpper)
    #uncomment this line below if you want
    cv2.imshow('filteredFrame', filteredFrame)

    # create a frame that uses filteredFrame to only show the
    # filtered sections of the original frame and black out the rest
    colorCutout =  cv2.bitwise_and(frame, frame, mask=filteredFrame)
    # show the resulting frame
    cv2.imshow('colorCutout', colorCutout)
    #/////////////////////////////

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
