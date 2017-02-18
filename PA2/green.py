import sys
import numpy as np
import cv2
import os

#####input file name#####
if len(sys.argv) != 2:
	print "Usage: python a1.py <in_file>"
	sys.exit()
in_file=sys.argv[1]

cap = cv2.VideoCapture(in_file)
if cap.isOpened()==False:
	print("Can not open the video")
	sys.exit()

ret, frame = cap.read()

b, g, r = cv2.split(frame)

cv2.imwrite('g.png', g)
cv2.imwrite('r.png', r)
cv2.imwrite('b.png', b)

