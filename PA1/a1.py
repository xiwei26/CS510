import sys
import numpy as np
import cv2

#####input and output file directories and names#####
if len(sys.argv) < 3:
	print "not enought arguments"
	sys.exit()
in_file=sys.argv[1]
print in_file
out_file=sys.argv[2] + '.avi'

#####read video and information#########
cap = cv2.VideoCapture(in_file)
#cap = cv2.VideoCapture('example1.mov')
if cap.isOpened()==False:
	print("Can not open the video")
	sys.exit()
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('../out.avi',fourcc, 30.0, (1280,720))
#out = cv2.VideoWriter('output.mov', -1, 30, (1280,720))
total_frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#print (total_duration)
fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))

codec = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))

#print(fps)

####user input#######
start_frame = int(raw_input("Please tell me the start frame (0-based): "))

end_frame = int(raw_input("Please tell me the last frame (0-based): "))

startX = int(raw_input("what's the object's start x: "))
startY = int(raw_input("what's the object's start y: "))
height = int(raw_input("what is the object's height: "))
width = int(raw_input("what is the object's width: "))

speed_in_x= float(raw_input("what's the velosity in x axis: ")) #can be double
speed_in_y = float(raw_input("what's the velosity in y axis: ")) #can be double

output_ht = int(raw_input("what is the output video's height: "))
output_wd = int(raw_input("what is the output video's width: "))

#start_frame_no=(start_frame/(total_duration*fps))
#print(start_frame_no)
#end_frame_no=(end_frame/(total_duration*fps))
#print(end_frame_no)
#cap.set(2, start_frame) 
#totalframe=end_frame-start_frame
#cap.set(7, duration) 

out = cv2.VideoWriter(out_file, codec, fps, (output_wd,output_ht))
if out.isOpened() == False:
	print("Cannot open output file!")
	sys.exit()

cap.set(1, start_frame)
for i in range (start_frame,end_frame + 1):

	currX = float(startX + speed_in_x * (i - start_frame) ) #Better this way if speed_in_x is a double
	currY = float(startY + speed_in_y * (i - start_frame) ) 
	ret, frame = cap.read()
	#a = np.matrix('1 2; 3 4')
	pts1 = np.float32([[currX, currY],[currX, currY+height-1],[currX+width-1, currY]])
	pts2 = np.float32([[0,0],[0,output_ht-1],[output_wd-1,0]])

	M = cv2.getAffineTransform(pts1,pts2)
	#cv2.rectangle(frame, (startX, startY), (startX + width, startY+ height), (0,0,255),2)
	dst = cv2.warpAffine(frame, M, (output_wd,output_ht))
	out.write(dst)
	#cv2.imshow('window-name',dst)
	#cv2.imwrite("frame%d.jpg" % count, frame)
	#count = count + 1
	#if cv2.waitKey(10) & 0xFF == ord('q'):
	#	break


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


#cv2.destroyAllWindows()
cap.release()
out.release()

#cap.release()
#cv2.destroyAllWindows()

