import cv2

cap = cv2.VideoCapture(0)

#/////////////////////////////

#we will be using this for the trackbar on change parameter
def nothing(x):
    pass


# Creating a window for later use
cv2.namedWindow('Control Panel')

# set the initial hue, saturation, and value
initialHue = 10
initialSat = 10
initialVal = 10

# set the initial hue, saturation, and value ranges
initialHR = 10
initialSR = 10
initialVR = 10



# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('Hue', 'Control Panel', initialHue, 180, nothing)  # default 0 205 255 69 8 12
cv2.createTrackbar('Sat', 'Control Panel', 205, 255, nothing)
cv2.createTrackbar('Val', 'Control Panel', 255, 255, nothing)
cv2.createTrackbar('Hrange', 'Control Panel', 69, 127, nothing)
cv2.createTrackbar('Srange', 'Control Panel', 69, 127, nothing)
cv2.createTrackbar('Vrange', 'Control Panel', 69, 127, nothing)



#/////////////////////////////

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

def findBiggestContour(mask):
    # create an array of the contours found in mask
    contoursArray = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    # only proceed if at least one contour was found
    if len(contoursArray) > 0:
        # find the biggest contour in the array of contours
        biggestCountour = max(contoursArray, key=cv2.contourArea)
        return biggestCountour

while (1):
    # save the image frame
    ret, frame = cap.read()


    #///////////////////////
    #use our function that create a new frame with the color filtered
    #we will call it mask
    mask = filterColor(frame)
    #this frame has to be a binary frame (black and white)

    # find the biggest contour
    biggestContour = findBiggestContour(mask)
    # find the rectangle bounding the biggest contour
    x, y, w, h = cv2.boundingRect(biggestContour)

    rectangleColor = (255, 0, 0) #(Blue,Green,Red) they go from 0-255
    # create a rectangle in the original frame around the biggest contour
    cv2.rectangle(frame, (x, y), (x + w, y + h), rectangleColor, 2)

    #choose the font for the text we will display
    font = cv2.FONT_HERSHEY_SIMPLEX

    # put the text of the values on the screen to the user can see them
    #cv2.putText(frame, "X: " + format(x), (0, 50), font, 0.6, (155, 250, 55), 2, cv2.LINE_AA)
    #cv2.putText(frame, "Y: " + format(y), (0, 100), font, 0.6, (155, 250, 55), 2, cv2.LINE_AA)

    # put the text and have it move! on the screen
    cv2.putText(frame, "X: " + format(x), (int(x), int(y)+50), font, 0.6, (155, 250, 55), 2, cv2.LINE_AA)
    cv2.putText(frame, "Y: " + format(y), (int(x), int(y)+100), font, 0.6, (155, 250, 55), 2, cv2.LINE_AA)

    # ///////////////////////
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
