import cv2

cap = cv2.VideoCapture(0)

#/////////////////////////////

#we will be using this for the trackbar on change parameter
def nothing(x):
    pass


# Creating a window for later use
cv2.namedWindow('Control Panel')

# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('Hue', 'Control Panel', 0, 180, nothing)  # default 0 205 255 69 8 12
cv2.createTrackbar('Sat', 'Control Panel', 205, 255, nothing)
cv2.createTrackbar('Val', 'Control Panel', 255, 255, nothing)
cv2.createTrackbar('Hrange', 'Control Panel', 69, 127, nothing)
cv2.createTrackbar('Srange', 'Control Panel', 69, 127, nothing)
cv2.createTrackbar('Vrange', 'Control Panel', 69, 127, nothing)
#----------------------------------------------------------------


def filterColor(frame):
    # create a new frame in
    # "Hue Saturation Value" or HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # /////////////////////////////

    # get info from track bar and apply to result
    hue = cv2.getTrackbarPos('Hue', 'Control Panel')
    sat = cv2.getTrackbarPos('Sat', 'Control Panel')
    val = cv2.getTrackbarPos('Val', 'Control Panel')
    hrange = cv2.getTrackbarPos('Hrange', 'Control Panel')
    srange = cv2.getTrackbarPos('Srange', 'Control Panel')
    vrange = cv2.getTrackbarPos('Vrange', 'Control Panel')


    #create a boundary for any color that gets update by trackbars
    colorLower = (hue - hrange, sat - srange, val - vrange)
    colorUpper = (hue + hrange, sat + srange, val + vrange)
    #( Hue 0-180, Saturation 0-255, Value 0-255 )
    #EXAMPLE
    #say the user moves trackbars set the hue at 10 and hrange to 10 then
    #the lower hue will be 10-10 = 0 and upper hue will be 10+10=20.

    # /////////////////////////////
    # create a filtered frame that only captures
    # objects with colors within the boundaries set by the trackbars
    filteredFrame = cv2.inRange(hsv, colorLower, colorUpper)
    #cv2.imshow('filteredFrame', filteredFrame)

    # create a frame that uses filteredFrame to only show the
    # filtered sections of the original frame and black out the rest
    colorCutout =  cv2.bitwise_and(frame, frame, mask=filteredFrame)
    # show the resulting frame
    cv2.imshow('colorCutout', colorCutout)

    return filteredFrame


while (1):
    # save the image frame
    ret, frame = cap.read()


    # ///////////////////////////////////////////////////////////////////

    #use our function that create a new frame with the color filtered
    #we will call it mask
    mask = filterColor(frame)
    mask = cv2.dilate(mask, None)
    mask = cv2.blur(mask, (21, 21))
    mask = cv2.dilate(mask, None)

    #mask = cv2.medianBlur(mask, 5)
    cv2.imshow('mask', mask)

    # combined = cv2.addWeighted(mask, 1.0, edges, 1, 0)
    # cv2.imshow('combined', combined)
    # create an array of the contours found in mask
    contoursArray = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    # only proceed if at least one contour was found
    if len(contoursArray) > 0:
        # sort the contours by area
        sortedCont = sorted(contoursArray, key=cv2.contourArea, reverse=True)[:3]
        # loop through the sorted contours
        for i in range(0,len(sortedCont)):
            countour = sortedCont[i]
            # find the x value, y value, width, and height of the rectangle
            # that bounds the contour
            x, y, w, h = cv2.boundingRect(countour)

            # establish a color for the rectangle
            rectangleColor = (255, i*85, 255-i*85) #(Blue,Green,Red) they go from 0-255

            # create a rectangle in the frame using the values from the contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangleColor, 2)
            cv2.putText(frame, "#:" + format(i), (int(x), int(y) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rectangleColor, 2, cv2.LINE_AA)
    # ///////////////////////////////////////////////////////////////////
    # show the rectangle in the image frame
    cv2.imshow('rectangle', frame)
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
