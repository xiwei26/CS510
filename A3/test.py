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
T = 25
kernel = np.ones((5,5),np.uint8)

for i in range(total_frame_num):
	ret, frame = cap.read()
	fgmask = fgbg.apply(frame)
	fgmask = ndimage.gaussian_filter(fgmask, sigma=l/(4.*n))
	label_im, nb_labels = ndimage.label(fgmask)
	print nb_labels
	cv2.imshow("xxx",label_im)

	filename = './result/' + str(i) + '.png'
	cv2.imwrite(filename,fgmask)
