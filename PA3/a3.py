import sys
import cv2
import numpy as np

##### read input file #####
if len(sys.argv) != 2:
	print "Usage: python a3.py <in_file>"
	sys.exit()

in_file = sys.argv[1]
cap = cv2.VideoCapture(in_file)
if cap.isOpened() == False:
	print("Can not open the video")
	sys.exit()
total_frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

### Background substraction ###
fgbg = cv2.BackgroundSubtractorMOG()
kernel = np.ones((5,5),np.uint8)

for i in range(total_frame_num):
	ret, frame = cap.read()
	if(i == 0):
		fgmask = fgbg.apply(frame)
	else:
		fgbg.apply(frame, fgmask, -1)

	closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

	ret, thresh = cv2.threshold(opening, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	### draw bounding rectangles ###
	for a in contours:
		x,y,w,h = cv2.boundingRect(a)
		if(w >= 20 and h >= 20):
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	print i

	filename = './result4/' + str(i) + '.png'
	cv2.imwrite(filename,frame)