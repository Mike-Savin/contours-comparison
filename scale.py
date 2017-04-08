import cv2, sys

# Could be used to resize images with openCV
# Has some optional command line provided arguments

def resize(image, scale=0.2):
	return cv2.resize(image, (0,0), fx=scale, fy=scale)

# Default hardcoded sources
sources = [
	'img/c/1.png',
	'img/c/3.png',
	'img/c/4.png',
	'img/c/5.png',
	'img/c/6.png',
	'img/c/7.png',
	'img/c/8.png',
	'img/c/9.png',
	'img/c/10.png',
	'img/c/paints/1.png',
	'img/c/paints/2.png',
	'img/c/paints/3.png',
	'img/c/paints/4.png',
	'img/c/_2.png'
]

if __name__ == '__main__':
	scale=0.2
	if len(sys.argv[1:]) > 2:
		sources, dist, scale = [sys.argv[1]], [sys.argv[2]], int(sys.argv[3])

	for index, src in zip(range(len(sources)), sources):
		print src
		image = cv2.imread(src)
		scaled = resize(image, scale)
		cv2.imwrite(sources[index] + '.scaled.png', scaled)