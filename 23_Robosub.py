import cv2
import numpy as np

cap = cv2.VideoCapture('Semi_Final_Run.mp4')
# i know this semi final run vid is 2 minutes and 20seconds
# 2*60+20 seconds = 140seconds


###-----------HISTOGRAM----------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
hsv_map = np.zeros((180, 256, 3), np.uint8)
hist, s = np.indices(hsv_map.shape[:2])
hsv_map[:, :, 0] = hist
hsv_map[:, :, 1] = s
hsv_map[:, :, 2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)

cv2.namedWindow('hist', 0)
hist_scale = 10

def set_scale(val):
    global hist_scale
    hist_scale = val

cv2.createTrackbar('scale', 'hist', hist_scale, 32, set_scale)
##------------------------------------------

divideFrameBy = 2
# save the image frame
ret, frame = cap.read()
# find the size
height, width, channel = frame.shape

print("width:", width, " height:", height, "channels:", channel)
print("Frames per Second: ", cap.get(cv2.CAP_PROP_FPS))

frameNumber = 0

# TOTALFRAMES = cap.get(cv2.CV_CAP_PROP_FRAME)
# TOTALFRAMES = 1000


durationOfVideo = 140  # 140 seconds
# now we need to calculate the frames
TOTALFRAMES = int(cap.get(cv2.CAP_PROP_FPS) * durationOfVideo)


# we will be using this for the trackbar on change parameter
def nothing(x):
    pass


# # Creating a window for later use
# cv2.namedWindow('Control Panel')

# Creating a window for later use
cv2.namedWindow('Main Panel')

# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
# cv2.createTrackbar('Hue', 'Control Panel', 0, 180, nothing)  # default 0 205 255 69 8 12
# cv2.createTrackbar('Sat', 'Control Panel', 205, 255, nothing)
# cv2.createTrackbar('Val', 'Control Panel', 255, 255, nothing)
# cv2.createTrackbar('Hrange', 'Control Panel', 69, 127, nothing)
# cv2.createTrackbar('Srange', 'Control Panel', 69, 127, nothing)
# cv2.createTrackbar('Vrange', 'Control Panel', 69, 127, nothing)

cv2.createTrackbar('frameNumber', 'Main Panel', frameNumber, TOTALFRAMES, nothing)  # default 0 205 255 69 8 12
cv2.createTrackbar('Hue', 'Main Panel', 0, 180, nothing)  # default 0 205 255 69 8 12
cv2.createTrackbar('Sat', 'Main Panel', 205, 255, nothing)
cv2.createTrackbar('Val', 'Main Panel', 255, 255, nothing)
cv2.createTrackbar('Hrange', 'Main Panel', 69, 127, nothing)
cv2.createTrackbar('Srange', 'Main Panel', 69, 127, nothing)
cv2.createTrackbar('Vrange', 'Main Panel', 69, 127, nothing)



# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None



def filterColor(frame):
    # create a new frame in
    # "Hue Saturation Value" or HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # /////////////////////////////

    # get info from track bar and apply to result
    # hue = cv2.getTrackbarPos('Hue', 'Control Panel')
    # sat = cv2.getTrackbarPos('Sat', 'Control Panel')
    # val = cv2.getTrackbarPos('Val', 'Control Panel')
    # hrange = cv2.getTrackbarPos('Hrange', 'Control Panel')
    # srange = cv2.getTrackbarPos('Srange', 'Control Panel')
    # vrange = cv2.getTrackbarPos('Vrange', 'Control Panel')

    hue = cv2.getTrackbarPos('Hue', 'Main Panel')
    sat = cv2.getTrackbarPos('Sat', 'Main Panel')
    val = cv2.getTrackbarPos('Val', 'Main Panel')
    hrange = cv2.getTrackbarPos('Hrange', 'Main Panel')
    srange = cv2.getTrackbarPos('Srange', 'Main Panel')
    vrange = cv2.getTrackbarPos('Vrange', 'Main Panel')

    # create a boundary for any color that gets update by trackbars
    colorLower = (hue - hrange, sat - srange, val - vrange)
    colorUpper = (hue + hrange, sat + srange, val + vrange)
    # ( Hue 0-180, Saturation 0-255, Value 0-255 )
    # EXAMPLE
    # say the user moves trackbars set the hue at 10 and hrange to 10 then
    # the lower hue will be 10-10 = 0 and upper hue will be 10+10=20.

    # /////////////////////////////

    filteredFrame = cv2.inRange(hsv, colorLower, colorUpper)
    # cv2.imshow('filteredFrame', filteredFrame)

    colorCutout = cv2.bitwise_and(frame, frame, mask=filteredFrame)
    cv2.imshow('colorCutout', colorCutout)

    return filteredFrame


while frameNumber < TOTALFRAMES:

    frameNumber = cv2.getTrackbarPos('frameNumber', 'Main Panel')
    frameNumber = frameNumber + 1
    cv2.setTrackbarPos('frameNumber', 'Main Panel', frameNumber)
    #frameNumber = cv2.getTrackbarPos('frameNumber', 'Main Panel')

    cap.set(1, frameNumber)

    ret, frame = cap.read()
    frameNormal = frame


    # resize image
    frame = cv2.resize(frame, (int(width / divideFrameBy), int(height / divideFrameBy)))

    # frame_1 = frame
    # frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    #
    # ret, frame_2 = cap.read()
    # frame_2 = cv2.resize(frame_2, (int(width / divideFrameBy), int(height / divideFrameBy)))
    # frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    # frameDelta = cv2.absdiff(frame_1, frame_2)
    # et, frameDelta = cv2.threshold(frameDelta, 40, 255, cv2.THRESH_BINARY)
    # cv2.imshow("movement", frameDelta)

    # ///////////////////////////////////////////////////////////////////

    # ---------------------------------------

    # use our function that create a new frame with the color filtered
    # we will call it mask
    mask = filterColor(frame)
    # mask = cv2.dilate(mask, None)
    # mask = cv2.blur(mask, (21, 21))

    # mask = cv2.medianBlur(mask, 5)
    # cv2.imshow('mask', mask)



    contoursArray = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
    # only proceed if at least one contour was found
    if len(contoursArray) > 0:
        # sort the list of contours from biggest to smallest limit it to 3
        sizeOfContourList = 3
        sortedCont = sorted(contoursArray, key=cv2.contourArea, reverse=True)[:sizeOfContourList]
        # go through all of the sorted contours and draw rectangles
        for i in range(0, len(sortedCont)):
            countour = sortedCont[i]
            x, y, w, h = cv2.boundingRect(countour)
            rectangleColor = (255, i * 85, 255 - i * 85)  # (Blue,Green,Red) they go from 0-255
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangleColor, 2)
            cv2.putText(frame, "#:" + format(i), (int(x), int(y) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rectangleColor,
                        2, cv2.LINE_AA)
    # ///////////////////////////////////////////////////////////////////

    #---------------
    small = cv2.pyrDown(frameNormal)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dark = hsv[..., 2] < 32
    hsv[dark] = 0
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    hist = np.clip(hist * 0.005 * hist_scale, 0, 1)
    vis = hsv_map * hist[:, :, np.newaxis] / 255.0
    cv2.imshow('hist', vis)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    gray = cv2.equalizeHist(gray)
    cv2.imshow('gray', gray)

    #ret,thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    #cv2.imshow('thresh', thresh)
    edges = cv2.Canny(gray, 100, 150)
    cv2.imshow('edges', edges)

    # ----------------

    # gray32 = np.float32(gray)
    # dst = cv2.cornerHarris(gray32, 10, 3, 0.05)

    # #corners
    # small[dst > 0.01 * dst.max()] = [255, 255, 255]
    # cv2.imshow('small', small)



    # show the image frame
    cv2.imshow('Main Panel', frame)

    # show the image frame and
    # wait for 1 milliseconds
    # to check if user click "q"
    KeyPressed = cv2.waitKey(1) & 0xFF
    if KeyPressed == ord('a'):
        frameNumber = cv2.getTrackbarPos('frameNumber', 'Main Panel')
        cv2.setTrackbarPos('frameNumber', 'Main Panel',frameNumber)

    if KeyPressed == ord('q'):
        break
# if cv2.waitKey(1) & 0xFF == ord('q'):
#    break
    frameNumber = frameNumber + 1

# if user clicks "q" then break out of the loop


# make sure to release the camera
cap.release()
# close all windows
cv2.destroyAllWindows()
