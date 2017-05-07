import sys
import numpy as np
import cv2
import os
import mosse
import common
import math

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from vgg16 import vgg16

from collections import Counter


'''
1. resizeImg() takes the frame and the feature window, producing a 224*224 image; if the input feature window is a rectangle,
	this function only takes the center part of the image
2. Put a4.py in the vgg folder 
3. Picked 5 still objects with highest probability, 5 lines in final result
4. Each tracking (not each object) is one result line in the final result
'''

def one_moving_object(obj,name):
	f = open('./result_one.txt', 'a')
	increment=0
	print(len(obj))

	if(len(obj)>4):
		increment =int(math.floor(len(obj)/4))
		first=0
		second=increment
		third=second+increment
		fourth=len(obj)-1
		collection=[first,second,third,fourth]
		for i in collection:
			time=math.ceil(obj[i][0]/30)+1
			x_tl, y_tl, x_br, y_br = obj[i][1]
			center_x1=(x_br+x_tl)/2
			center_y1=(y_br+y_tl)/2

			x_ntl, y_ntl, x_nbr, y_nbr = obj[i+1][1]
			center_x2=(x_nbr+x_ntl)/2
			center_y2=(y_nbr+y_ntl)/2

			if(center_x2>center_x1):
			#moving_east=True
				f.write("A "+name+" is moving towards east at "+str(time)+"seconds. ")
			if(center_x2<center_x1):
			#moving_west=True
				f.write("A "+name+" is moving towards west at "+str(time)+"seconds. ")
			if(center_y2>center_y1):
			#moving_south=True
				f.write("A "+name+" is moving towards south at "+str(time)+"seconds. ")
			if(center_y2<center_y1):
			#moving_north=True
				f.write("A "+name+" is moving towards north at "+str(time)+"seconds. ")
	else:
		for i in range(len(obj)-1):
			time=math.ceil(obj[i+1][0]/30)+1
			print("here0")
			x_tl, y_tl, x_br, y_br = obj[i][1]
			center_x1=(x_br+x_tl)/2
			center_y1=(y_br+y_tl)/2
			print(center_x1,center_y1)
			x_ntl, y_ntl, x_nbr, y_nbr = obj[i+1][1]
			center_x2=(x_nbr+x_ntl)/2
			center_y2=(y_nbr+y_ntl)/2
			print(center_x2,center_y2)
			if(center_x2>center_x1):
			#moving_east=True
				f.write("A "+name+" is moving towards east at "+str(time)+"seconds. ")
			if(center_x2<center_x1):
			#moving_west=True
				f.write("A "+name+" is moving towards west at "+str(time)+"seconds. ")
			if(center_y2>center_y1):
			#moving_south=True
				f.write("A "+name+" is moving towards south at "+str(time)+"seconds. ")
			if(center_y2<center_y1):
			#moving_north=True
				f.write("A "+name+" is moving towards north at "+str(time)+"seconds. ")
	f.write("\n")
	f.close()
	print('Writing Completed')			

def two_moving_object(tracker1, name1, tracker2, name2):
	f = open('./result_two.txt', 'w')
	sentence = ""
	verb = ""
	time = 0
	distance = 0
	common_frames = []
	for i in range(len(tracker1)):
		for j in range(len(tracker2)):
			if(tracker1[i][0] == tracker2[j][0]):
				common_frames.append((i,j))

	print common_frames
	for i in range(len(common_frames)):
		if(i == 0 or i == len(common_frames)-1 or 
			i == math.floor(len(common_frames)/4) or 
			i == math.floor(len(common_frames)/4*3)):
			sentence = ""
			print i
			index1 = common_frames[i][0]
			index2 = common_frames[i][1]
			time = math.ceil(tracker1[index1][0]/30)+1

			# convert coordinates to center point for object1
			x1_tl,y1_tl,x1_br,y1_br = tracker1[index1][1]
			x1 = (x1_tl + x1_br)/2
			y1 = (y1_tl + y1_br)/2
		
			# convert coordinates to center point for object2
			x2_tl,y2_tl,x2_br,y2_br = tracker2[index2][1]
			x2 = (x2_tl + x2_br)/2
			y2 = (y2_tl + y2_br)/2
		
			# compute distance and determine their relatvie position
			new_distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

			# generate sentence
			if(new_distance < distance):
				verb = " moved toward "
			elif(new_distance > distance):
				verb = " moved away from "
		
			# generate sentece for each frame
			sentence += name1 + verb + name2 + " at " + str(time) + " second."
			sentence += "\n"
			f.write(sentence)

			# update distance
			distance = new_distance

	f.write("\n")
	f.close()
	print('Writing Completed')	


def check_overlap_moving(obj, t_objs):
	flag = False
	#print t_objs
	for i in range(len(t_objs)):
		dx = min(obj[2], t_objs[i][-1][1][2]) - max(obj[0], t_objs[i][-1][1][0])
		dy = min(obj[3], t_objs[i][-1][1][3]) - max(obj[1], t_objs[i][-1][1][1])
		if (dx>=0) and (dy>=0):
			t_obj_width = t_objs[i][-1][1][2]-t_objs[i][-1][1][0]
			t_obj_height = t_objs[i][-1][1][3]-t_objs[i][-1][1][1]
			area1 = float(t_obj_width*t_obj_height)
			obj_width = obj[2] - obj[0]
			obj_height = obj[3] - obj[1]
			area2 = float(obj_width * obj_height)
			overlap_rate = max(dx*dy/area1, dx*dy/area2)
			if(overlap_rate > 0.5):
				flag = True
				break
	return flag

# obj(upleft_x, upleft_y, lowright_x, lowright_y)
def find_moving_objects(labels):
	moving_objects = []
	num_labels = labels[0]
	stats = labels[2]
	for i in range(num_labels):
		x = stats[i, cv2.CC_STAT_LEFT]
		y = stats[i, cv2.CC_STAT_TOP]
		w = stats[i, cv2.CC_STAT_WIDTH]
		h = stats[i, cv2.CC_STAT_HEIGHT]
		area = stats[i, cv2.CC_STAT_AREA]

		if(w > 15 and w < 500):
			if(h > 40):
				moving_objects.append((x,y,x+w,y+h))
	return moving_objects

def write_tracks(tracks):
	f = open('result.csv', 'w')
	f.write('Track #, Frame #, x(upper left), y(upper right), x(lower right), y(lower right)\n')
	for i in range(len(tracks)):
		for j in range(len(tracks[i])):
			f.write(str(i+1)+',')
			f.write(str(tracks[i][j][0]))
			for t in range(4):
				if tracks[i][j][1][t]<0:
					f.write(',0')
				else:
					f.write(',' + str(tracks[i][j][1][t]))
			f.write('\n')
	f.close()

def resizeImg(frame, center_x, center_y, width, height):
	### take the center square if input is a rectangle ###
	if width > height:
		width = height 
	if height > width:
		height = width 

	tl_x=int(center_x-(width/2)) #topleft_x
	tl_y=int(center_y-(height/2)) #topleft_y

	### affine transformation ###
	pts1 = np.float32([[tl_x, tl_y], [tl_x, tl_y + height], [tl_x + width, tl_y]])
	pts2 = np.float32([[0, 0], [0, 223],[223, 0]])
	M = cv2.getAffineTransform(pts1,pts2)

	output_ht = 224
	output_wd = 224
	return cv2.warpAffine(frame, M, (output_wd,output_ht))

def checkOverlap_static(x1,y1,r1,x2,y2,r2):
	'''
	Function copied from PA2
	'''
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
	'''
	Function copied from PA2
	'''
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#If there are less than 30 key points, lower hessian value and try again
	while 1:
		#surf = cv2.SURF(float(hessian))
		surf = cv2.xfeatures2d.SURF_create(hessian)
		kp, des = surf.detectAndCompute(gray,None)
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
				if checkOverlap_static(b[0],b[1],b[2]/2,x,y,diameter/2) == True:
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

def shrink(frame, n):
	'''
	Function copied from PA2
	'''
	####  Shrink image (both width and height) by n fold
	rows, cols, ch = frame.shape
	pts1 = np.float32([[0,0], [0,rows-1], [cols-1, 0]])
	pts2 = np.float32([[0,0], [0,(rows-1)//n], [(cols-1)//n, 0]])
	M_shrink = cv2.getAffineTransform(pts1, pts2)
	M_enlarge = cv2.getAffineTransform(pts2, pts1)
	frame_shrink = cv2. warpAffine(frame, M_shrink, (cols, rows))
	return (frame_shrink, M_enlarge)

def write_result(still_obj, tracking_objects):
	print('Writing result to csv')
	f = open('./result.csv', 'w')
	f.write('Type, Frame #, Frame #, x(upper left), y(upper left), object_label, %\n')

	### find 5 best still objects ###
	if len(still_obj) > 5:
		act_levls = [still_obj[i][5] for i in range(len(still_obj))]
		best_still_index = sorted(range(len(act_levls)), key=lambda k: act_levls[k])[-5:]
	else:
		best_still_index = range(len(still_obj))
	
	# Write still objects
	for j in range(len(best_still_index)):
		f.write('Still')
		s_obj =  still_obj[best_still_index[j]]
		for ele in s_obj:
			f.write(',' + str(ele))
		f.write('\n')

	### write moving object 
	#percent_matching = [0 for i in range(len(moving_objects))]
	for tracked in tracking_objects:
		first_frame = tracked[0][0]
		x_tl, y_tl, x_br, y_br = tracked[0][1]
		object_label = tracked[0][2]
		last_frame = tracked[-1][0]
		object_label_cnt = 0
		for mov_obj in tracked:
			if mov_obj[2] == object_label:
				object_label_cnt += 1
		percent_matching = object_label_cnt *1.0 / len(tracked) #percent of frames matching the label
		f.write('Moving, %d, %d, %d, %d, %s, %.4f\n' \
				%(first_frame, last_frame, x_tl, y_tl, object_label.replace(',', '/'), percent_matching))

	f.close()
	print('Writing Completed')

def most_frequent_class(target):
	classes = []
	for ea_fm in target:
		classes.append(ea_fm[2])
	count = Counter(classes)
	nm = count.most_common()[0][0]
	return nm.split(',')[0].lower()

#####input file name#####
if len(sys.argv) != 2:
	print("Usage: python a1.py <in_file>")
	sys.exit()
in_file=sys.argv[1]

#isGaussian = raw_input("Are you going to run gaussian variation analysis (y or n): ")
#isGaussian = True if isGaussian == 'y' else False
isGaussian = True

#isResize = raw_input("Are you going to resize image (y or n): ")
#isResize = True if isResize == 'y' else False
isResize = False

#####read video information#####

cap = cv2.VideoCapture(in_file)
if cap.isOpened()==False:
	print("Can not open the video")
	sys.exit()

total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# total_frame_num = 100
#####processing video#####
coordinatesList = []
hessian = 5000
ksize = 15 ### parameter for gaussian blur
M_enlarge = np.array([0]) ### used for resizing attention window when frame size is shrinked
fold = 2 ### shrinking fold

#sum_sz = 0 ### for calculating average attention window size
#large_sz = 0  ### for calculating maximum attention window size

#paramters for static features
still_obj = []

#parameters for tracking
trackers = []
tracking_objects = []
tracking_objects_NoFrame = []   # record number of frames for each tracked object

### vgg object, copied from vgg16.py ###
sess = tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(images, 'vgg16_weights.npz', sess)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

#iterate each frame
for i in range (total_frame_num):
	print('Processing Frame %d' %i)
	ret, frame = cap.read()
	print('-----Moving Object')
	####  Process moving objects/features ####
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	### find moving objects ###
	if(i == 0):
		fgmask = fgbg.apply(frame)
	else:
		fgbg.apply(frame, fgmask, -1)

	labels = cv2.connectedComponentsWithStats(fgmask, 8, cv2.CV_32S);
	moving_objects = find_moving_objects(labels)
#	for obj in moving_objects:
#		cv2.rectangle(gray_frame, (obj[0],obj[1]), (obj[2],obj[3]), (0,255,0),2)
	### track objects ###
	if i >= 5:
		track_no = -1

		## first track already identified objects
		for j in range(len(trackers)):
			track_no = j
			trackers[j].update(gray_frame)	
			if trackers[j].good:
				x1,y1,x2,y2 = trackers[j].getRect()
				center_x=(x1+x2)/2
				center_y=(y1+y2)/2
				height=y2-y1
				width=x2-x1

				img_temp2=resizeImg(frame, center_x, center_y, width, height)
				image_stack2 = [img_temp2] 
				probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack2})
				preds = np.argmax(probs, axis=1)
				for index, p in enumerate(preds):
					#print("Prediction #%d: %s; Probability: %f"%(i, class_names[p], probs[index, p]))
					tracking_objects[j].append((i,(x1,y1,x2,y2), #frame#, obj_coordinate
												class_names[p])) #object_label
					tracking_objects_NoFrame[j] = (i - tracking_objects[j][0][0] + 1)

				trackers[j].draw_state(gray_frame)

		## next add newly identified objects
		for obj in moving_objects:
			track_no += 1
			if not check_overlap_moving(obj,tracking_objects):
				new_tracker = mosse.MOSSE(gray_frame, obj)
					#new_tracker.draw_state(gray_frame)
				x1,y1,x2,y2 = new_tracker.getRect()
				center_x=(x1+x2)/2
				center_y=(y1+y2)/2
				height=y2-y1
				width=x2-x1

				trackers.append(new_tracker)
				
				img_temp2=resizeImg(frame, center_x, center_y, width, height)
				image_stack2 = [img_temp2] 
				probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack2})
				preds = np.argmax(probs, axis=1)
				for index, p in enumerate(preds):
					#print("Prediction #%d: %s; Probability: %f"%(i, class_names[p], probs[index, p]))
					tracking_objects.append([(i, obj, class_names[p])])
					tracking_objects_NoFrame.append(1)
				new_tracker.draw_state(frame, (track_no+1))
	#cv2.imwrite('./moving/' + str(i) + '_moving.jpg', frame)
	print('---Frame %d Done' %i)

#write_result(still_obj, tracking_objects)
#print(still_obj)					



####### Summarize result  ###############
sorted_ind = np.argsort(tracking_objects_NoFrame)[::-1]
targets = []
names = []
for i in range(4):
	targets.append(tracking_objects[sorted_ind[i]])
	names.append(most_frequent_class(targets[i]))
	one_moving_object(targets[i], names[i])

obj_list = []
for i in range(4):
	if(len(targets[i]) > 100):
		obj_list.append(i)

two_moving_object(targets[obj_list[0]],names[obj_list[0]],targets[obj_list[1]],names[obj_list[1]])
#for x in names:
#	print(x)
#for x in targets:
#	print(x)

cap.release()
cv2.destroyAllWindows()
