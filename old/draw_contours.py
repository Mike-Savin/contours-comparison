import cv2
import numpy as np
from matplotlib import pyplot as plt

def auto_canny(img, sigma=0.33):
	v = np.median(img)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	return cv2.Canny(img, lower, upper)


img_loaded = cv2.imread('img/1.jpg')
img_gray_filtered = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2GRAY)

img_noise_removed = cv2.GaussianBlur(img_gray_filtered, (3, 3), 0)
#img_noise_removed = cv2.bilateralFilter(img_noise_removed, 11, 17, 17)
#img_noise_removed = img_gray_filtered

ret, thresh = cv2.threshold(img_gray_filtered,127, 255, 0)
smth, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img_with_contours = img_loaded.copy()

cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)

# convolute with proper kernels
laplacian = cv2.Laplacian(img_noise_removed, cv2.CV_64F)
sobelx = cv2.Sobel(img_noise_removed, cv2.CV_64F, 1, 0, ksize=5)  # x
sobely = cv2.Sobel(img_noise_removed, cv2.CV_64F, 0, 1, ksize=5)  # y

img_auto_canny = auto_canny(img_noise_removed)
img_wide_canny = cv2.Canny(img_noise_removed, 10, 200)
img_tight_canny = cv2.Canny(img_noise_removed, 225, 250)


plt.subplot(3,3,1), plt.imshow(cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,2), plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Find contours'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,3), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


plt.subplot(3,3,6), plt.imshow(img_auto_canny, cmap='gray')
plt.title('Canny (auto)'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,7), plt.imshow(img_wide_canny, cmap='gray')
plt.title('Canny (wide)'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,8), plt.imshow(img_tight_canny, cmap='gray')
plt.title('Canny (tight)'), plt.xticks([]), plt.yticks([])

plt.show()