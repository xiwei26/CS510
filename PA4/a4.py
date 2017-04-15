import sys
import numpy as np
import cv2
import os
import mosse
import common

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from vgg16 import vgg16


'''
1. resizeImg() takes the frame and the feature window, producing a 224*224 image; if the input feature window is a rectangle,
	this function only takes the center part of the image
2. Put a4.py in the vgg folder 
3. Picked 5 still objects with highest probability, 5 lines in final result
4. Each tracking (not each object) is one result line in the final result
'''
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
		best_still_index = [0,1,2,3,4]
		p_thresh_index = 0 #used to find the index with lowest probability in the best 5 objects
		for i in range(5,len(still_obj)):

			#find the lowest probability and the index in the best 5 objects
			p_thresh = still_obj[best_still_index[0]][5]
			for j in range(5): 
				if p_thresh > still_obj[best_still_index[j]][5]:
					p_thresh = still_obj[best_still_index[j]][5]
					p_thresh_index = j
			#update to find the best 5 objects
			if still_obj[i][5] > p_thresh:
				best_still_index[p_thresh_index] = i

	# Write still objects
	for j in range(5):
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

#total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_frame_num = 20
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

### vgg object, copied from vgg16.py ###
sess = tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(images, 'vgg16_weights.npz', sess)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

#iterate each frame
for i in range (total_frame_num):
	print('Processing Frame %d' %i)
	ret, frame = cap.read()

	print('-----Static Object')
	### Process static objects/features ###
	if not isGaussian:
		if not isResize:
			frame_kp = frame
		else:
			frame_kp, M_enlarge = shrink(frame, fold)
	else:
		frame_gaus = cv2.GaussianBlur(frame, (ksize, ksize), 0)
		if not isResize:
			frame_kp = frame_gaus
		else:
			frame_kp, M_enlarge = shrink(frame_gaus, fold)

	total_kp, hessian = kp_detect(frame_kp, coordinatesList, hessian)
	img_tmp = resizeImg(frame, coordinatesList[-1][0], coordinatesList[-1][1], coordinatesList[-1][2], coordinatesList[-1][2])
	#img_tmp2 =resizeImg(frame, tracking_objects[-1][0], tracking_objects[-1][1], tracking_objects[-1][2], tracking_objects[-1][2])
	image_stack = [img_tmp] #if image_stack == None else np.append(image_stack, [img_tmp], axis = 0)
	cv2.imwrite('./result/' + str(i) + '_static.jpg', img_tmp)
	
	### Predict static object class ###
	probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})
	preds = np.argmax(probs, axis=1)
	#for index, p in enumerate(preds):
	#    print("Prediction #%d: %s; Probability: %f"%(i, class_names[p], probs[index, p]))
	for index, p in enumerate(preds):
		still_obj.append((i, i, #frame#, frame#
						coordinatesList[-1][0] - coordinatesList[-1][2] // 2, #x_upper_left
						coordinatesList[-1][1] - coordinatesList[-1][2] // 2, #y_upper_left
						class_names[p].replace(',', '/'), # object_label, replace ',' as /
						probs[index, p]))  # activation_level

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
				#trackers[j].draw_state(gray_frame)

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
#				new_tracker.draw_state(frame, (track_no+1))
	print('-----Frame %d Done' %i)

write_result(still_obj, tracking_objects)
#print(still_obj)					




	#print "Frame#",i,",keypoint:", coordinatesList[-1]
	#writeKP(frame, coordinatesList[-1][0], coordinatesList[-1][1], coordinatesList[-1][2], './result/', i, M_enlarge)

	#sz = coordinatesList[-1][2] * fold if isResize else coordinatesList[-1][2]
	#sum_sz += sz
	#large_sz = sz if sz > large_sz else large_sz 

#print 'Gaussian Blur' if isGaussian else 'No Gaussian Blur'
#print 'Resize image' if isResize else 'No Resizing'
#print 'Average attention window size is: ', sum_sz/total_frame_num
#print 'Largest attention window size is: ', large_sz

cap.release()
cv2.destroyAllWindows()