import sys
import numpy as np
import cv2
import os

#####input and output file directories and names#####
if len(sys.argv) != 2:
	print "Usage: python a1.py <in_file>"
	sys.exit()
in_file=sys.argv[1]
#out_file=sys.argv[2]

#if os.path.isfile(out_file):
#	os.remove(out_file)

#####user input#####


#startX = int(raw_input("what's the object's start x: "))
#startY = int(raw_input("what's the object's start y: "))
#height = int(raw_input("what is the object's height: "))
#width = int(raw_input("what is the object's width: "))

#speed_in_x= float(raw_input("what's the velosity in x axis: ")) #can be double
#speed_in_y = float(raw_input("what's the velosity in y axis: ")) #can be double

#output_ht = int(raw_input("what is the output video's height: "))
#output_wd = int(raw_input("what is the output video's width: ")) 


#####read video and information#####
cap = cv2.VideoCapture(in_file)
if cap.isOpened()==False:
	print("Can not open the video")
	sys.exit()

total_frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

coordinates = []


#####processing video#####
for i in range (0, total_frame_num):
	ret, frame = cap.read()
	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	surf = cv2.SURF(20000)
	kp = surf.detect(gray,None)
	
	x = 0;
	y = 0;
	total_kp = len(kp)
	diameter = 0;
	count =0;
	for a in range(0,total_kp-1):
		x = int(kp[a].pt[0]);
		y = int(kp[a].pt[1]);
		#print "check:", a
		#print x
		#print y
		
		if (x,y) in coordinates:
			#print "(x,y already in coordinates)"
			continue
		else:
			if count<30:
				coordinates.append((x,y))
				diameter=kp[a].size
				count=count+1
				break
			else:
				coordinates.pop(0)
				diameter=kp[a].size
				coordinates.append((x,y))
				break

	#print "coordinate len: ",len(coordinates)
	print "final coordinate :",x,y
	

	#print len(kp)
	#print size

	tl_x=int(x-(diameter/2))
	tl_y=int(y-(diameter/2))
	br_x=int(x+(diameter/2))
	br_y=int(y+(diameter/2))

	cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (0,0,255),2)

	filename="img"+str(i)+".png"
	cv2.imwrite(filename,frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

		#currX = int(startX + speed_in_x * (i - start_frame) ) #Better this way if speed_in_x is a double
		#currY = int(startY + speed_in_y * (i - start_frame) ) 
		
		#pts1 = np.float32([[currX, currY],[currX, currY+height-1],[currX+width-1, currY]])
		#pts2 = np.float32([[0,0],[0,output_ht-1],[output_wd-1,0]])

		#M = cv2.getAffineTransform(pts1,pts2)
		#cv2.rectangle(frame, (currX, currY), (currX + width, currY+ height), (0,0,255),2)
		#dst = cv2.warpAffine(frame, M, (output_wd,output_ht))
		#out.write(dst)
		#		#sift=cv2.SIFT()
		#kp=sift.detect(gray,None)
		#frame=cv2.drawKeypoints(gray,kp)
		#cv2.imwrite('sift_keypoints.jpg',frame)

		#if cv2.waitKey(10) & 0xFF == ord('q'):
		#	break



cap.release()
#out.release()
cv2.destroyAllWindows()
