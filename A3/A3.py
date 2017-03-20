import sys
import cv2
import numpy as np
import scipy
from scipy import ndimage

##### read input file #####
if len(sys.argv) != 2:
	print "Usage: python a3.py <in_file>"
	sys.exit()

in_file = sys.argv[1]
cap = cv2.VideoCapture(in_file)
if cap.isOpened() == False:
	print("Can not open the video")
	sys.exit()
total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_file = open("resultR", 'w')
### Background substraction ###
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
n=10
l=256
kernel = np.ones((5,5),np.uint8)
information = []
for i in range(total_frame_num):
	ret, frame = cap.read()
	if(i == 0):
		fgmask = fgbg.apply(frame)
	else:
		fgbg.apply(frame, fgmask, -1)


	#opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	
	output = cv2.connectedComponentsWithStats(fgmask, 8, cv2.CV_32S);
	num_labels = output[0]
	stats = output[2]
	index =0 
	for j in range(num_labels):
		x = stats[j, cv2.CC_STAT_LEFT]
		y = stats[j, cv2.CC_STAT_TOP]
		w = stats[j, cv2.CC_STAT_WIDTH]
		h = stats[j, cv2.CC_STAT_HEIGHT]
		area = stats[j, cv2.CC_STAT_AREA]
		
		lr_x=x+w;
		lr_y=y+h;
		if(w > 22 and w < 500):
			if(h > 40):
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				if(i==1):
					out_file.write(str(index)+","+str(i)+","+str(x)+","+str(y)+","+str(lr_x)+","+str(lr_y)+"\n")
					information.append(index,i,x,y,lr_x,lr_y)
					index=index+1
				else:
					#start tracking

				print w,h,area
	print i

	filename = './result/' + str(i) + '.png'
	cv2.imwrite(filename,frame)
