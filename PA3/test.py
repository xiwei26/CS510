#import scipy as sp
from scipy import  ndimage
import numpy as np
import cv2
import sys
'''
img = cv2.imread('128.png', 0)
#n_comp, labels = sp.csgraph.connected_components(img)
blobs_labels, nb_labels = ndimage.label(img)#, background=0)
print nb_labels
#for i in range(len(all_labels)):
#	if blobs_labels[i].any():
#		print (i, blobs_labels[i])
#cv2.imshow('a', blobs_labels)
cv2.imwrite('test.png',blobs_labels)
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''
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
#total_frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
total_frame_num = 381

fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
codec = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))

out = cv2.VideoWriter('test_short.mov', codec, fps, (output_wd,output_ht))
if out.isOpened() == False:
	print("Cannot open output file!")
	sys.exit()
### Background substraction ###
#fgbg = cv2.BackgroundSubtractorMOG()

for i in range(total_frame_num):
	ret, frame = cap.read()
	#if i >= 103 and i <= 151:
	print i
	#fgmask = fgbg.apply(frame)
	#filename = './tracking/' + str(i) + '_ori.png'
	#	cv2.imwrite(filename,frame)
	out.write(frame)
### Connected components ###
out.release()
