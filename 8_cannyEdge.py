import cv2

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

# Creating a window for later use
cv2.namedWindow('CannyEdge')

# Creating track bar
# cv.CreateTrackbar(trackbarName, windowName, value, count, onChange)  None
cv2.createTrackbar('threshold-1', 'CannyEdge', 130, 1000, nothing)
cv2.createTrackbar('threshold-2', 'CannyEdge', 80, 1000, nothing)

while (1):
    # Take each frame
    ret, frame = cap.read()

    # get info from the trackbars
    th1 = cv2.getTrackbarPos('threshold-1', 'CannyEdge')
    th2 = cv2.getTrackbarPos('threshold-2', 'CannyEdge')

    edges = cv2.Canny(frame, th1, th2)

    cv2.imshow('edges', edges)

    keyPressed = cv2.waitKey(1) & 0xFF

    if keyPressed == ord('q'):
        break

cv2.destroyAllWindows()
