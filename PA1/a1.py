import sys
import numpy as np
import cv2


if len(sys.argv)!=3:
	print "not enought arguments"

in_file=sys.argv[1]
out_file=sys.argv[2]







cap = cv2.VideoCapture('/Users/xiwei/Desktop/pa1/example1.mov')
if cap.isOpened()==False:
	print("Can not find the video")

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('../out.avi',fourcc, 30.0, (1280,720))

out = cv2.VideoWriter('output.mov', -1, 30, (1280,720))

total_duration = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print (total_duration)

fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
print(fps)



start_frame = int(raw_input("Please tell me the start frame: "))
end_frame = int(raw_input("Please tell me the last frame:"))
startX = int(raw_input("what's the object's start x: "))
startY = int(raw_input("what's the object's start y: "))
height = int(raw_input("what is the object's height: "))
width = int(raw_input("what is the object's width: "))
speed_in_x= int(raw_input("what's the velosity in x of the object: "))
speed_in_y = int(raw_input("what's the velosity in y of the object: "))

start_frame_no=(start_frame/(total_duration*fps))
print(start_frame_no)
end_frame_no=(end_frame/(total_duration*fps))
print(end_frame_no)
#cap.set(2, start_frame) 
#totalframe=end_frame-start_frame
#cap.set(7, duration) 

cap.set(1, start_frame)

for num in range (start_frame,end_frame):
	ret,frame = cap.read()

	#a = np.matrix('1 2; 3 4')


	pts1 = np.float32([[startX,startY],[startX,startY+height],[startX+width,startY]])
	pts2 = np.float32([[0,0],[0,720],[1280,0]])
	M = cv2.getAffineTransform(pts1,pts2)
	#cv2.rectangle(frame, (startX, startY), (startX + width, startY+ height), (0,0,255),2)


	dst = cv2.warpAffine(frame,M,(1280,720))





	
	startX = startX + speed_in_x
	startY = startY + speed_in_y
	out.write(dst)
	cv2.imshow('window-name',dst)
	#cv2.imwrite("frame%d.jpg" % count, frame)
	#count = count + 1
	





	if cv2.waitKey(10) & 0xFF == ord('q'):
		break


#while(cap.isOpened()):
#    ret, frame = cap.read()
#
#    cv2.imshow('capture',frame)
#   if cv2.waitKey(100) & 0xFF == ord('q'):
#        break

# take first frame of the video
#ret,frame = cap.read()

# setup initial location of window
#r,h,c,w = startRow,height,startRow,width  # simply hardcoded the values
#track_window = (c,r,w,h)


cv2.destroyAllWindows()
cap.release()
out.release()

#cap.release()
#cv2.destroyAllWindows()

print "end"
