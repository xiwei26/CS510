import sys
import numpy as np
import cv2
import os
import mosse
import common

import tensorflow as tf
import scipy
from imagenet_classes import class_names
from vgg16 import vgg16

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

def resizeImg(frame, center_x, center_y, width, height):
	### take the center square if input is a rectangle ###
	if width > height:
		width = height 
	if height > width:
		height = width 

	tl_x=int(center_x-(width/2)) #topleft_x
	tl_y=int(center_y-(height/2)) #topleft_y

	### affine transformation ###
	pts1 = np.float32([[tl_x, tl_y], [tl_x, tl_y+diameter], [tl_x+diameter, tl_y]])
	pts2 = np.float32([[0, 0], [0, 223],[223, 0]])
	M = cv2.getAffineTransform(pts1,pts2)

	output_ht = 224
	output_wd = 224
	return cv2.warpAffine(frame, M, (output_wd,output_ht))

def check_overlap_static(x1,y1,r1,x2,y2,r2):
	### return true if overlaps > 50%
	### x1 y1 -> last frame keypoint
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
		surf = cv2.xfeatures2d.SURF_create(hessian)
		kp, des = surf.detectAndCompute(gray,None)
		total_kp = len(kp)
		if total_kp > 30:
			break
		else:
			hessian -= 1000

	for i in range(total_kp):
		x = int(kp[i].pt[0])
		y = int(kp[i].pt[1])
		diameter = kp[i].size
		# check if (x,y) already in coordinates list
		if (x,y,diameter) not in coordinatesList:
			for j in coordinatesList:
				if check_overlap_static(j[0],j[1],j[2]/2,x,y,diameter/2) == True:
					break
			else:
				if len(coordinatesList) < 30:
					coordinatesList.append((x,y,diameter))
					break
				else:
					coordinatesList.pop(0)
					coordinatesList.append((x,y,diameter))
					break
	return (x,y,diameter,hessian)

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

	#write moving objects
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

if __name__ == "__main__":
	#####read input file#####
	if len(sys.argv) != 2:
		print("Usage: python a4.py <in_file>")
		sys.exit()

	in_file = sys.argv[1]
	cap = cv2.VideoCapture(in_file)

	if cap.isOpened()==False:
		print("Can not open the video")
		sys.exit()

	total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# total_frame_num = 20

	#paramters for static features
	ksize = 15
	hessian = 10000
	coordinatesList = []
	still_obj = []

	#parameters for tracking
	trackers = []
	tracking_objects = []
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

	### vgg object, copied from vgg16.py ###
	sess = tf.Session()
	images = tf.placeholder(tf.float32, [None, 224, 224, 3])
	vgg = vgg16(images, 'vgg16_weights.npz', sess)
	

	for i in range(total_frame_num):
		ret, frame = cap.read()
		print('Processing Frame %d' %i)
		print('-----Static Object')
		frame_gaus = cv2.GaussianBlur(frame, (ksize, ksize), 0)
		x,y,diameter,hessian = kp_detect(frame_gaus, coordinatesList, hessian)
		img_tmp = resizeImg(frame, x, y, diameter, diameter)
		image_stack = [img_tmp]
 		# cv2.imwrite('./static/' + str(i) + '.jpg', img_tmp)
 		
 		tl_x=int(x-(diameter/2)) #topleft_x
		tl_y=int(y-(diameter/2)) #topleft_y
		br_x=int(x+(diameter/2)) #botright_x
		br_y=int(y+(diameter/2)) #botright_y
 		cv2.rectangle(frame, (tl_x,tl_y),(br_x,br_y), (0,255,0), 2)

 		### Predict static object class ###
		probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})
		preds = np.argmax(probs, axis=1)

		# for index, p in enumerate(preds):
	 #   		print("Prediction #%d: %s; Probability: %f"%(i, class_names[p], probs[index, p]))
		for index, p in enumerate(preds):
			still_obj.append((i, i, #frame#, frame#
						x - diameter//2, #x_upper_left
						y - diameter//2, #y_upper_left
						class_names[p].replace(',', '/'), # object_label, replace ',' as /
						probs[index, p]))  # activation_level

		print('-----Moving Object')
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		### find moving objects ###
		if(i == 0):
			fgmask = fgbg.apply(frame)
		else:
			fgbg.apply(frame, fgmask, -1)

		labels = cv2.connectedComponentsWithStats(fgmask, 8, cv2.CV_32S);
		moving_objects = find_moving_objects(labels)

		### track objects ###
		if i >= 5:
			track_no = -1

			## first track already identified objects
			for j in range(len(trackers)):
				track_no = j
				trackers[j].update(gray_frame)	
				if trackers[j].good:
					x,y,w,h = trackers[j].getRect()
					#trackers[j].draw_state(gray_frame)
					img_temp2 = resizeImg(frame,x,y,w,h)
					image_stack2 = [img_temp2]
					# cv2.imwrite('./moving/' + str(i) + "_" + str(j) + '.jpg', img_temp2)
					probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack2})
					preds = np.argmax(probs, axis=1)
					trackers[j].draw_state(frame, (track_no+1))
					for index, p in enumerate(preds):
						
						tracking_objects[j].append((i,(x-w//2,y-h//2,x+w//2,y+h//2),class_names[p]))

			## next add newly identified objects
			for obj in moving_objects:
				track_no += 1
				if not check_overlap_moving(obj,tracking_objects):
					new_tracker = mosse.MOSSE(gray_frame, obj)
					x,y,w,h = new_tracker.getRect()
					trackers.append(new_tracker)

					img_temp2 = resizeImg(frame,x,y,w,h)
					image_stack2 = [img_temp2]
					# cv2.imwrite('./moving/' + str(i) + '.jpg', img_temp2)
					probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack2})
					preds = np.argmax(probs, axis=1)
					for index, p in enumerate(preds):
						tracking_objects.append([(i, obj, class_names[p])])

					new_tracker.draw_state(frame, (track_no+1))
					
					
		
		cv2.imwrite('./raw/' + str(i) + '.jpg', frame)
		print('-----Frame %d Done' %i)

	write_result(still_obj, tracking_objects)
	cap.release()
	cv2.destroyAllWindows()

















