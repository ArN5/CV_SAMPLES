#!/usr/bin/env python

'''
Video histogram sample to show live histogram of video
Keys:
    ESC    - exit
    https://github.com/opencv/opencv/blob/master/samples/python/color_histogram.py
'''

import numpy as np
import cv2

# built-in modules
import sys

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cap = cv2.VideoCapture(0)

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

while (1):
    flag, frame = cap.read()

    #frame = cv2.equalizeHist(frame)
    cv2.imshow('frame', frame)
    small = cv2.pyrDown(frame)


    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dark = hsv[..., 2] < 32
    hsv[dark] = 0
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    hist = np.clip(hist * 0.005 * hist_scale, 0, 1)
    vis = hsv_map * hist[:, :, np.newaxis] / 255.0
    cv2.imshow('hist', vis)


    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    #cv2.imshow('equ', equ)
    cl1 = clahe.apply(gray)
    #cv2.imshow('cl1', cl1)
    res = np.hstack((equ, cl1))
    cv2.imshow('both', res)

    ch = cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()
