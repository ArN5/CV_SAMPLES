
import cv2
import numpy as np
#based on this tutorial
#https://docs.opencv.org/3.4.1/d1/d5c/tutorial_py_kmeans_opencv.html

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('Semi_Final_Run.mp4')
# Initiate STAR detector
orb = cv2.ORB_create()
def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

cv2.createTrackbar('value 1', 'Control Panel', 4, 10, nothing)
cv2.createTrackbar('value 2', 'Control Panel', 10, 1000, nothing)
cv2.createTrackbar('value 3', 'Control Panel', 10, 1000, nothing)

while (1):

    val_1 = cv2.getTrackbarPos('value 1', 'Control Panel')
    val_2 = cv2.getTrackbarPos('value 2', 'Control Panel')
    val_3 = cv2.getTrackbarPos('value 3', 'Control Panel')

    _, frame = cap.read()

    height, width, layers = frame.shape

    # comment this line if you want the fullsize window
    img = cv2.resize(frame, (int(width / 2), int(height / 2)))

    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = val_1+1

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]

    res2 = res.reshape((img.shape))

    cv2.imshow('res2', res2)
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