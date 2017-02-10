import sys
import numpy as np
import cv2
import os

#####input and output file directories and names#####
if len(sys.argv) != 3:
	print "Usage: python a1.py <in_file> <out_file>"
	sys.exit()
in_file=sys.argv[1]
out_file=sys.argv[2]

if os.path.isfile(out_file):
	os.remove(out_file)

#####user input#####
start_frame = int(raw_input("Please tell me the start frame (0-based): "))
end_frame = int(raw_input("Please tell me the last frame (0-based): "))

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
fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
codec = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))



#####processing video#####
for i in range (start_frame, end_frame + 1):
	ret, frame = cap.read()
	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	surf = cv2.SURF(20000)
	kp = surf.detect(gray,None)
	
	x = kp[0].pt[0];
	y = kp[0].pt[1];
	size = kp[0].size;
	print len(kp)
	print size

	tl_x=int(x-(size/2))
	tl_y=int(y-(size/2))
	br_x=int(x+(size/2))
	br_y=int(y+(size/2))

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
