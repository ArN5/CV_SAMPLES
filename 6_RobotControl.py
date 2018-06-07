import cv2

cap = cv2.VideoCapture(0)

#/////////////////////////////

#we will be using this for the trackbar on change parameter
def nothing(x):
    pass


# Creating a window for later use
cv2.namedWindow('Control Panel')

starbucksGreen = [80,111,78,16,49,30]
brightOrange = [8,255,208,6,115,55]

# choose the color you want
#colorIwant = starbucksGreen
colorIwant = brightOrange

#Hue (color),
#Saturation (concentration of hue)
#Value (concentration of black or white)
# set the initial hue, saturation, and value
initialHue = colorIwant[0]
initialSat = colorIwant[1]
initialVal = colorIwant[2]

# set the initial hue, saturation, and value range
initialHR = colorIwant[3]
initialSR = colorIwant[4]
initialVR = colorIwant[5]



# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('Hue', 'Control Panel', initialHue, 180, nothing)  # default 0 205 255 69 8 12
cv2.createTrackbar('Sat', 'Control Panel', initialSat, 255, nothing)
cv2.createTrackbar('Val', 'Control Panel', initialVal, 255, nothing)
cv2.createTrackbar('Hrange', 'Control Panel', initialHR, 127, nothing)
cv2.createTrackbar('Srange', 'Control Panel', initialSR, 127, nothing)
cv2.createTrackbar('Vrange', 'Control Panel', initialVR, 127, nothing)


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
    cv2.imshow('colorCutout', colorCutout)

    return filteredFrame

#function that finds the biggest "object found" in the list of coordinates of "colored objects" found
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

    biggestContour = findBiggestContour(mask)
    #find the center coordinates and width and height of a bounding rectangle around the countour
    x, y, w, h = cv2.boundingRect(biggestContour)

    rectangleColor = (255, 0, 0) #(Blue,Green,Red) they go from 0-255
    #so right now the rectangle is blue

    #this line will actually draw the rectangle onto the "frame" frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), rectangleColor, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # put the text and have it move! on the screen
    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    cv2.putText(frame, "X: " + format(x), (int(x)-60, int(y)+50), font, 0.6, (155, 250, 55), 2, cv2.LINE_AA)
    cv2.putText(frame, "Y: " + format(y), (int(x)-60, int(y)+70), font, 0.6, (155, 255, 155), 2, cv2.LINE_AA)
    cv2.putText(frame, "W: " + format(w), (int(x)-60, int(y)+90), font, 0.6, (215, 250, 55), 2, cv2.LINE_AA)
    cv2.putText(frame, "H: " + format(h), (int(x)-60, int(y)+110), font, 0.6, (155, 250, 155), 2, cv2.LINE_AA)
    # ///////////////////////
    cv2.imshow('frame', frame)



    #--------------------------------------------------------------------------------------------
    #first find the size of the image frame in pixels.
    #example my frame may be 640 pixels in width
    height, width, layers = frame.shape

    middleXpixel = width / 2
    # pixel tolerance we will allow
    tolerance = 20

    #Logic for robot (Remember camera is mirrored)
    if(x>(middleXpixel+tolerance)):
        #example: middle point 640/2+20= 340
        print("go to the left side")
    elif(x<(middleXpixel-tolerance)):
        # example: middle point 640/2-20 = 280
        print("go to the right side")
    else:
        #this will only happen if between 280-340
        print("go to the middle")



    #--------------------------------------------------------------------------------------------


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
