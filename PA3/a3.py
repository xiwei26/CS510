import sys
import cv2
import numpy as np
import mosse
import common

def check_overlap(obj, t_objs):
	flag = False
	for i in range(len(tracking_objects)):
		dx = min(obj[2], t_objs[i][1][2]) - max(obj[0], t_objs[i][1][0])
		dy = min(obj[3], t_objs[i][1][3]) - max(obj[1], t_objs[i][1][1])
		if (dx>=0) and (dy>=0):
			t_obj_width = t_objs[i][1][2]-t_objs[i][1][0]
			t_obj_height = t_objs[i][1][3]-t_objs[i][1][1]
			area = float(t_obj_width*t_obj_height)
			overlap_rate = dx*dy/area
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

if __name__ == '__main__':

	### read input file ###
	if len(sys.argv) != 2:
		print "Usage: python a3.py <in_file>"
		sys.exit()

	in_file = sys.argv[1]
	cap = cv2.VideoCapture(in_file)
	if cap.isOpened() == False:
		print("Can not open the video")
		sys.exit()
	total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	### initialize MOG model ###
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

	trackers = []
	tracking_objects = []

	for i in range(total_frame_num):
		print i
		ret, frame = cap.read()
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		### find moving objects ###
		if(i == 0):
			fgmask = fgbg.apply(frame)
		else:
			fgbg.apply(frame, fgmask, -1)

		labels = cv2.connectedComponentsWithStats(fgmask, 8, cv2.CV_32S);
		moving_objects = find_moving_objects(labels)
		'''
		for obj in moving_objects:
			cv2.rectangle(gray_frame, (obj[0],obj[1]), (obj[2],obj[3]), (0,255,0),2)
		'''
		### track objects ###
		if i >= 5:
			for j in range(len(trackers)):
				trackers[j].update(gray_frame)	
				if trackers[j].good:
					x1,y1,x2,y2 = trackers[j].getRect()
					trackers[j].draw_state(gray_frame)
					tracking_objects[j]=((i,(x1,y1,x2,y2),trackers[j].psr))
					print "%d,%d,%d,%d,%d,%d" % (j,i,x1,y1,x2,y2)

			for obj in moving_objects:
				if not check_overlap(obj,tracking_objects):
					new_tracker = mosse.MOSSE(gray_frame, obj)
					new_tracker.draw_state(gray_frame)
					trackers.append(new_tracker)
					tracking_objects.append((i, obj, new_tracker.psr))

	
		filename = './result/' + str(i) + '.png'
		cv2.imwrite(filename,gray_frame)
		









