import cv2
import numpy as np
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

#src – input array
#dst – output array of the same size and type as src


# Initialize camera, ORB detector, and BFMatcher object
cap = cv2.VideoCapture(0)
orb = cv2.ORB_create()
bf = cv2.BFMatcher()
bfH = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
#bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


# Define the image we want to find
#img1 = cv2.imread('iphone.png', 0)
img1 = cv2.imread('lotsIphones.png', 0)

#resize image

divideFrameBy = 4

height, width = img1.shape
img1 = cv2.resize(img1, (int(width / divideFrameBy), int(height / divideFrameBy)))

# Get keypoints and descriptors for image
kp1, des1 = orb.detectAndCompute(img1, None)


def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('Control Panel')

cv2.createTrackbar('numberOfMatches', 'Control Panel', 2, 1000, nothing)
cv2.createTrackbar('ratio Test', 'Control Panel', 7, 1000, nothing)
cv2.createTrackbar('value 3', 'Control Panel', 1, 1000, nothing)
cv2.createTrackbar('value 4', 'Control Panel', 2, 1000, nothing)



while (1):

    numberOfMatches = cv2.getTrackbarPos('numberOfMatches', 'Control Panel')
    ratioTest = cv2.getTrackbarPos('ratio Test', 'Control Panel')
    val_3 = cv2.getTrackbarPos('value 3', 'Control Panel')
    val_4 = cv2.getTrackbarPos('value 4', 'Control Panel')

    # Take each frame
    _, frame = cap.read()
    height, width, layers = frame.shape

    frame = cv2.resize(frame, (int(width / divideFrameBy), int(height / divideFrameBy)))

    #turn to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get keypoints and descriptors for video frame
    kp2, des2 = orb.detectAndCompute(gray, None)

    # Match descriptors
    matches = bfH.match(des1, des2)
    '''
    matches.distance - Distance between descriptors. The lower, the better it is.
    matches.trainIdx - Index of the descriptor in train descriptors
    matches.queryIdx - Index of the descriptor in query descriptors
    matches.imgIdx - Index of the train image.
    '''
    #sort that list of "matches" by distance from smallest to biggest
    matches = sorted( matches , key=lambda x: x.distance)

    #bf.match return only a list of single objects

    # Draw first matches which wll be the first smallest distances
    orbMatches = cv2.drawMatches(img1, kp1, gray, kp2, matches[:numberOfMatches], None, flags=2)

    # Display result
    cv2.imshow("Orb Matches", orbMatches)


    #KNN MATCHER
    matches = bf.knnMatch(des1, des2, k=2)

    # # Apply ratio test
    good = []

    #https://blog.csdn.net/wc781708249/article/details/78528617
    for m, n in matches:
        if m.distance < (1-1/(ratioTest+1)) * n.distance:
            good.append([m])
    knnMatches = cv2.drawMatchesKnn(img1, kp1, gray, kp2, good[:numberOfMatches], None, flags=2)
    #img4 = cv2.drawMatchesKnn(img1, kp1, gray, kp2, good, None,singlePointColor=(0,0,255) ,flags=2)

    #img4 = cv2.drawMatchesKnn(img1, kp1, gray, kp2, matches[:val_4], None, flags=2)

    cv2.imshow("knn-matcher", knnMatches)


    # Break on key press
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break  # Destroy all windows
cv2.destroyAllWindows()
