import numpy as np  # math libraries
import cv2  # opencv itself


def nothing(x):
    pass


thresholdLower = (0, 0, 0)
thresholdUpper = (255, 255, 255)

# Creating a window for later use
cv2.namedWindow('Control Panel')

# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('hue', 'Control Panel', 68, 180, nothing)  # default 0 205 255 69 8 12
cv2.createTrackbar('sat', 'Control Panel', 255, 255, nothing)
cv2.createTrackbar('val', 'Control Panel', 93, 255, nothing)
cv2.createTrackbar('Hue Range', 'Control Panel', 44, 180, nothing)
cv2.createTrackbar('Sat Range', 'Control Panel', 70, 127, nothing)
cv2.createTrackbar('Val Range', 'Control Panel', 70, 127, nothing)

cap = cv2.VideoCapture(0)


def trackRectangle(mask, frame):
    # contours,hierarchy = cv2.findContours(mask, 1, 2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            area = cv2.contourArea(cnt)
            rows, cols = mask.shape

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # if the rectangle formed has a height bigger than the width
            # then the ratio will depend on how many times bigger the height is with the width
            # so in this case the rectangle I had, was 3 times bigger on on side than the other side
            # YOU NEED TO adjust the ratio by measuring the spinning rectangle and find the ratio

            # example since one leg is 3 times bigger than the other
            # other leg is 1/3 the size of the big one 1/3=0.333


            ratioHW = float(1.0 * h / w)

            if ratioHW < 0.31:
                print("horizontal---------")
            elif ratioHW > 3.0:
                print("vertical ||||||||||")
            else:
                pass

            # create rect object
            rect = cv2.minAreaRect(cnt)

            # ---
            # center = rect[0]
            # centerX = int(rect[0][0])
            # centerY = int(rect[0][1])
            # rectWidth = rect [1][0]
            # rectHeight = rect [1][1]
            angle = rect[2]
            print("x", x, "y", y, "w", w, "h", h, " ratio", ratioHW, " angle", angle)

            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # red colored box
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    return frame


# pass in the orginal frame along with the masked threshold frame
def rotateRectangle(mask, frame):
    # used this tutorial
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    #frame = frame
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)

        rows, cols = mask.shape
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        rect = cv2.minAreaRect(cnt)
        center = rect[0]
        angle = rect[2]
        print("center ", rect[0])  # create a rotation matrix
        rot = cv2.getRotationMatrix2D(center, angle + 90, 0.5)
        box = cv2.boxPoints(rect)
        print("Angle", angle)

        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (255, 255, 255), 2)

        nonRotatedFrame = frame
        cv2.putText(nonRotatedFrame, format(rect[0]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(nonRotatedFrame, format(rect[2]), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow("non rotated frame", nonRotatedFrame)
        frame = cv2.warpAffine(frame, rot, (rows, cols))
        # M = cv2.getPerspectiveTransform(cx, cy)

        # dst = cv2.warpPerspective(mask,M,(300,300))

        return frame

while True:

    # get info from track bar and appy to result
    h = cv2.getTrackbarPos('hue', 'Control Panel')
    s = cv2.getTrackbarPos('sat', 'Control Panel')
    v = cv2.getTrackbarPos('val', 'Control Panel')
    hr = cv2.getTrackbarPos('Hue Range', 'Control Panel')
    sr = cv2.getTrackbarPos('Sat Range', 'Control Panel')
    vr = cv2.getTrackbarPos('Val Range', 'Control Panel')

    thresholdLower = (h - hr, s - sr, v - vr)
    thresholdUpper = (h + hr, s + sr, v + vr)

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, thresholdLower, thresholdUpper)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    finalFrame = trackRectangle(mask, frame)

    # wait for keyboard to be pressed store the value as key
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    # now display the image to the user
    cv2.imshow("res", res)
    cv2.imshow("finalFrame", finalFrame)

cv2.destroyAllWindows()
