import cv2
import numpy as np
import sys

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

for i in range(total_frame_num):
	ret, frame = cap.read()
	fgmask = fgbg.apply(frame)
	filename = './result/' + str(i) + '.png'
	cv2.imwrite(filename,fgmask)