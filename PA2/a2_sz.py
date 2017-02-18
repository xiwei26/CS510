import sys
import numpy as np
import cv2
import os

def writeKP(frame, x, y, diameter, folder, img_seq):
	#### Write figures with key points labels to files ####

	#### Box coordinates ####
	tl_x=int(x-(diameter/2))
	tl_y=int(y-(diameter/2))
	br_x=int(x+(diameter/2))
	br_y=int(y+(diameter/2))

	cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (0,0,255),2)

	filename = folder + str(img_seq) +".png"
	cv2.imwrite(filename,frame)

def writeKP_resize(frame, x, y, diameter, folder, img_seq, M_enlarge):
	#### Write figures with key points labels to files ####

	#### Box coordinates ####
	tl_x=int(x-(diameter/2))
	tl_y=int(y-(diameter/2))
	br_x=int(x+(diameter/2))
	br_y=int(y+(diameter/2))


	top_left_enlarge = np.dot(M_enlarge, np.array([[tl_x],[tl_y],[1]]))
	bottom_right_enlarge = np.dot(M_enlarge, np.array([[br_x],[br_y],[1]]))
	cv2.rectangle(frame, tuple(top_left_enlarge.astype(int).flatten()),
				 tuple(bottom_right_enlarge.astype(int).flatten()), (0,0,255),2)

	filename = folder + str(img_seq) +".png"
	cv2.imwrite(filename,frame)

def checkOverlap(x1,y1,r1,x2,y2,r2):
	####return true if overlaps > 50%
	#### x1 y1 -> last frame keypoint
	
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

def kp_detect(frame, coordinatesList, hessian):
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#If there are less than 30 key points, lower hessian value and try again
	while 1:
		surf = cv2.SURF(float(hessian))
		kp = surf.detect(gray,None)
		total_kp = len(kp)
		if total_kp > 100:
			break
		else:
			hessian -= 1000
	#print "-------------------"
	for a in range(total_kp):
		x = int(kp[a].pt[0])
		y = int(kp[a].pt[1])
		diameter = kp[a].size
		#print (a, x,y,diameter)
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
	#print "-------------------"
	return (total_kp, hessian)

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

#total_frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
total_frame_num = 50
#####processing video#####
coordinatesList = []
hessian = 5000

for i in range (total_frame_num):

	ret, frame = cap.read()

	### Shrink image
	rows, cols, ch = frame.shape
	pts1 = np.float32([[0,0], [0,rows-1], [cols-1, 0]])
	pts2 = np.float32([[0,0], [0,(rows-1)//2], [(cols-1)//2, 0]])
	M_shrink = cv2.getAffineTransform(pts1, pts2)
	M_enlarge = cv2.getAffineTransform(pts2, pts1)
	frame_shrink = cv2. warpAffine(frame, M_shrink, (cols, rows))
	#print M_shrink

	total_kp, hessian = kp_detect(frame_shrink, coordinatesList, hessian)

	#print "coordinate len: ",len(coordinates)
	print "Frame#",i,",keypoint:", coordinatesList[-1]
	#print total_kp
	writeKP_resize(frame, coordinatesList[-1][0], coordinatesList[-1][1], coordinatesList[-1][2], './result/',i, M_enlarge)

	#writeKP(frame, coordinatesList[-1][0], coordinatesList[-1][1], coordinatesList[-1][2], './result/',i)

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()