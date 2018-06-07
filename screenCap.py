import numpy as np
from PIL import ImageGrab
# did pip install pillow
import cv2
import imutils

# function that grabs a segment
# of my screen in order to grab video
# from slows computer down, a lot
def screenCap(x,y,width,height):
    # this grabs the image from the screen
    # bbox specifies specific region (bbox= x,y,width,height)
    img = ImageGrab.grab(bbox=(x, y, width, height))

    # this is very important or else the colors will not match
    img = img.convert('RGB')
    # convert the image into a numpy array
    img_np = np.array(img)

    h, w, channel_ = img_np.shape
    # resize the image array
    frame = imutils.resize(img_np, width)

    # important for color matching turns RGB into BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    return frame




while (True):

    frame  = screenCap(100, 190, 500, 500)

    cv2.imshow('frame', frame)

    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()