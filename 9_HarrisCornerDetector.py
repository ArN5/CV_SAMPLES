import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('value 1', 'Control Panel', 4, 255, nothing)
cv2.createTrackbar('value 2', 'Control Panel', 10, 1000, nothing)
cv2.createTrackbar('value 3', 'Control Panel', 10, 1000, nothing)
cv2.createTrackbar('value 4', 'Control Panel', 10, 1000, nothing)

while (1):
    # get info from trackbars
    val_1 = cv2.getTrackbarPos('value 1', 'Control Panel')
    val_2 = cv2.getTrackbarPos('value 2', 'Control Panel')
    val_3 = cv2.getTrackbarPos('value 3', 'Control Panel')
    val_4 = cv2.getTrackbarPos('value 4', 'Control Panel')

    _, frame = cap.read()

    height, width, layers = frame.shape

    # comment this line if you want the fullsize window
    img = cv2.resize(frame, (int(width / 2), int(height / 2)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, val_1+1, 3, 1/(val_2+1))

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    cv2.imshow('dst', dst)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 1/(val_3+1) * dst.max()] = [0, 0, 255]

    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    cv2.imshow('dst2', dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1/(1+val_4))
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    cv2.imshow('res', res)

    cv2.imshow('img', img)

    key = cv2.waitKey(1)
    if key & 0xff == 27:
        break
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    # if the 'q' key is pressed, stop the loop
    if key == ord("Q"):
        break

cv2.destroyAllWindows()
