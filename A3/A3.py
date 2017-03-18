import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np

import scipy
from scipy import ndimage

##### input file name #####
if len(sys.argv) != 2:
	print "Usage: python a1.py <in_file>"
	sys.exit()
in_file=sys.argv[1]

##### read video information #####
cap = cv2.VideoCapture(in_file)
if cap.isOpened()==False:
	print("Can not open the video")
	sys.exit()
total_frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#total_frame_num = 50

### Background substraction ###
fgbg = cv2.BackgroundSubtractorMOG()

n=10
l=256
kernel = np.ones((5,5),np.uint8)

for i in range(total_frame_num):
	ret, frame = cap.read()
	fgmask = fgbg.apply(frame)
	#opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	ret, thresh = cv2.threshold(closing, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for a in contours:
    # Approximates rectangles
		x,y,w,h = cv2.boundingRect(a)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)




	filename = './result/' + str(i) + '.png'
	cv2.imwrite(filename,closing)
