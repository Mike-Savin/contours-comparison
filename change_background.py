import cv2
import numpy as np
from matplotlib import pyplot as plt

# Could be used to remove background from picture with contour
# or to change the contour color

image = cv2.imread('img/c/2.png')
image[np.where((image == [0, 0, 0]).all(axis = 2))] = [255, 255, 255]

# cv2.imshow('img', image)
# cv2.waitKey(0)
cv2.imwrite('img/c/_2.png', image)