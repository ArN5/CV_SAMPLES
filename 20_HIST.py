import cv2

cap = cv2.VideoCapture(0)


while (1):
    # grab the current frame
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    '''
    IMAGES: source image of type uint8 or float32. it should be given in as a list, ie, [gray_img].
    CHANNELS: it is also given in as a list []. 
        It the index of channel for which we calculate histogram.
        For example, if input is grayscale image, its value is [0].
        For color image, you can pass [0],[1] or [2] to 
        calculate histogram of blue,green or red channel, respectively.
    MASK: mask image. To find histogram of full image, 
        it is set as None. However, if we want to get histogram
        of specific region of image, we should create a mask image for that and give it as mask.
    HISTSIZE: this represents our BIN count. 
    Need to be given in []. For full scale, we pass [256].
    RANGES: Normally, it is [0,256].
    
    cv2.calcHist(images=,channels=, mask=,histSize=,ranges=,hist=,accumulate=)
    channels = [0,1] because we need to process both H and S plane.
    bins = [180,256] 180 for H plane and 256 for S plane.
    range = [0,180,0,256] Hue value lies between 0 and 180 & Saturation lies between 0 and 256.
    https://docs.opencv.org/3.1.0/dd/d0d/tutorial_py_2d_histogram.html
    '''

    histHSV = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.imshow("histHSV", histHSV)

    histGRY = cv2.calcHist([gray], [0], None, [256], [0, 256])

    cv2.imshow("histGRY", histGRY)

    key = cv2.waitKey(1)

    if key & 0xff == 27:
        break





