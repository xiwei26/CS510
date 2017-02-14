import sys
import numpy as np
import cv2
import os

#return true if overlaps > 50%
def checkOverlap(x1,y1,x2,y2,radius):
	dx = min(x1+radius, x2+radius) - max(x1-radius, x2-radius)
	dy = min(y1+radius, y2+radius) - max(y1-radius, y2-radius)
	if (dx>=0) and (dy>=0):
		area = 4*radius*radius
		overlap_rate = dx*dy/float(area)
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
for i in range (0, total_frame_num):

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	surf = cv2.SURF(20000)
	kp = surf.detect(gray,None)

	x = 0
	y = 0
	total_kp = len(kp)
	diameter = 0

	# reset keypoint list every second
	if(i % 30 == 0):
		coordinatesList = []
		diameterList = []

	for a in range(0,total_kp-1):
		x = int(kp[a].pt[0])
		y = int(kp[a].pt[1])
		diameter = kp[a].size
		
		# check if (x,y) already in coordinates list
		if (x,y) not in coordinatesList:
			# check if at the same scale
			if diameter not in diameterList:
				coordinatesList.append((x,y))
				diameterList.append(diameter)
				break
			
			else:
				# get index of (x,y) which at the same scale
				overlap_flag = False
				for b in range(len(diameterList)):
					if diameterList[b] == diameter:
						if checkOverlap(coordinatesList[b][0],coordinatesList[b][1],x,y,diameter/2) == True:
							overlap_flag = True
							break
				if overlap_flag == False:
					coordinatesList.append((x,y))
					diameterList.append(diameter)
					break


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