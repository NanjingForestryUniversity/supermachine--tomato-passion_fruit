# -*- coding: utf-8 -*-
# @Time    : 2024/6/26 下午6:15
# @Author  : TG
# @File    : t.py
# @Software: PyCharm
import cv2
import numpy as np


def nothing(x):
    pass


# Create a window
cv2.namedWindow('Green Pixels Selector')

# Create trackbars for color change
cv2.createTrackbar('Lower Hue', 'Green Pixels Selector', 0, 255, nothing)
cv2.createTrackbar('Lower Sat', 'Green Pixels Selector', 100, 255, nothing)
cv2.createTrackbar('Lower Val', 'Green Pixels Selector', 0, 255, nothing)
cv2.createTrackbar('Upper Hue', 'Green Pixels Selector', 60, 255, nothing)
cv2.createTrackbar('Upper Sat', 'Green Pixels Selector', 180, 255, nothing)
cv2.createTrackbar('Upper Val', 'Green Pixels Selector', 60, 255, nothing)

# Load image
image = cv2.imread(r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\tg\test\23.bmp')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

while (True):
    # Get current positions of the trackbars
    lh = cv2.getTrackbarPos('Lower Hue', 'Green Pixels Selector')
    ls = cv2.getTrackbarPos('Lower Sat', 'Green Pixels Selector')
    lv = cv2.getTrackbarPos('Lower Val', 'Green Pixels Selector')
    uh = cv2.getTrackbarPos('Upper Hue', 'Green Pixels Selector')
    us = cv2.getTrackbarPos('Upper Sat', 'Green Pixels Selector')
    uv = cv2.getTrackbarPos('Upper Val', 'Green Pixels Selector')

    # Define the HSV range for green
    lower_green = np.array([lh, ls, lv])
    upper_green = np.array([uh, us, uv])

    # Convert the image to HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Create the mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Convert result to BGR for display
    res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('Green Pixels Selector', res_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the window
cv2.destroyAllWindows()
