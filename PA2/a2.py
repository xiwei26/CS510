import sys
import numpy as np
import cv2
import os

#return true if overlaps > 50%
# x1 y1 -> last frame keypoint
def checkOverlap(x1,y1,r1,x2,y2,r2):
	dx = min(x1+r1, x2+r2) - max(x1-r1, x2-r2)
	dy = min(y1+r1, y2+r2) - max(y1-r1, y2-r2)
	if (dx>=0) and (dy>=0):
		area = float(4*r1*r1)
		overlap_rate = dx*dy/area
		if(overlap_rate > 0.5):
			return True
		else:
			return False
	else:
		return False


#####input file name#####
if len(sys.argv) != 2:
	print "Usage: python a1.py <in_file>"
	sys.exit()
in_file=sys.argv[1]

#####read video information#####
cap = cv2.VideoCapture(in_file)
if cap.isOpened()==False:
	print("Can not open the video")
	sys.exit()

total_frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

#####processing video#####
coordinatesList = []

for i in range (0, total_frame_num):

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	surf = cv2.SURF(20000)
	kp = surf.detect(gray,None)

	x = 0
	y = 0
	total_kp = len(kp)
	diameter = 0

	for a in range(0,total_kp):
		x = int(kp[a].pt[0])
		y = int(kp[a].pt[1])
		diameter = kp[a].size
		
		# check if (x,y) already in coordinates list
		if (x,y,diameter) not in coordinatesList:
			for b in coordinatesList:
				if checkOverlap(b[0],b[1],b[2]/2,x,y,diameter/2) == True:
					break
			else:
				if len(coordinatesList) < 30:
					coordinatesList.append((x,y,diameter))
					break
				else:
					coordinatesList.pop(0)
					coordinatesList.append((x,y,diameter))
					break


			# check if at the same scale
			

	#print "coordinate len: ",len(coordinates)
	print "Frame#",i,",keypoint:",x,y,diameter
	

	#print len(kp)
	#print size

	tl_x=int(x-(diameter/2))
	tl_y=int(y-(diameter/2))
	br_x=int(x+(diameter/2))
	br_y=int(y+(diameter/2))

	cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (0,0,255),2)

	filename = str(i)+".png"
	cv2.imwrite(filename,frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()