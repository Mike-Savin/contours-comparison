import cv2
import numpy as np
from matplotlib import pyplot as plt

# Script used to compare contour with a set of others using openCV
# Outputs the result to the plots

# Needs some refactoring like the images structure
sources_to_compare = [
	'img/c/1.png.scaled.png',
	'img/c/3.png.scaled.png',
	'img/c/4.png.scaled.png',
	'img/c/5.png.scaled.png',
	'img/c/6.png.scaled.png',
	'img/c/7.png.scaled.png',
	'img/c/8.png.scaled.png',
	'img/c/9.png.scaled.png',
	'img/c/10.png.scaled.png',
	'img/c/paints/1.png.scaled.png',
	'img/c/paints/2.png.scaled.png',
	'img/c/paints/3.png.scaled.png',
	'img/c/paints/4.png.scaled.png'
]

reference = cv2.imread('img/c/_2.png.scaled.png')

def get_contours(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(img_gray, 127, 255,0)
	smth,contours,hierarchy = cv2.findContours(thresh, 2, 1)
	return contours[0]

def get_cnt_extreme_points(cnt):
	leftmost   = list(cnt[cnt[:,:,0].argmin()][0])
	rightmost  = list(cnt[cnt[:,:,0].argmax()][0])
	topmost    = list(cnt[cnt[:,:,1].argmin()][0])
	bottommost = list(cnt[cnt[:,:,1].argmax()][0])
	return leftmost + rightmost + topmost + bottommost

def get_cnt_perimeter(cnt):
	return cv2.arcLength(cnt, cv2.isContourConvex(cnt))

def get_cnt_area(cnt):
	M = cv2.moments(cnt)
	return M['m00']

def get_cnt_centroid(cnt):
	M = cv2.moments(cnt)
	x = int(M['m10'] / M['m00'])
	y = int(M['m01'] / M['m00'])
	return (x, y)

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape)[:2] / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
	return result

def compare(img1, img2, shapes_weight=50, centroid_weight=0.1,
	area_weight=0.0001, perimeter_weight=0.01, extreme_points_weight=0.1):
	cnt1 = get_contours(img1)
	cnt2 = get_contours(img2)

	centroid_difference = sum([abs(get_cnt_centroid(cnt1)[i] - get_cnt_centroid(cnt2)[i])
		for i in range(2)])

	area_difference = abs(get_cnt_area(cnt1) - get_cnt_area(cnt2))
	perimeter_difference = abs(get_cnt_perimeter(cnt1) - get_cnt_perimeter(cnt2))

	extreme_points_difference = sum([abs(get_cnt_extreme_points(cnt1)[i] -\
		get_cnt_extreme_points(cnt2)[i]) for i in range(4)])

	shapes_difference = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
	result = [
		area_difference           * area_weight,
		shapes_difference         * shapes_weight,
		centroid_difference       * centroid_weight,
		perimeter_difference      * perimeter_weight,
		extreme_points_difference * extreme_points_weight
	]
	# Return weighted sum of contour similarity signs
	return sum(result)

# Plotting the result
if __name__ == '__main__':
	index = 1
	for src in sources_to_compare:
		image = cv2.imread(src)
		comparison_result = compare(reference, image)

		plt.subplot(2, len(sources_to_compare) / 2 + 1, index), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		plt.title(comparison_result), plt.xticks([]), plt.yticks([])

		index += 1

	plt.subplot(2, len(sources_to_compare) / 2 + 1, len(sources_to_compare) + 1), plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
	plt.title('Reference'), plt.xticks([]), plt.yticks([])

	plt.show()