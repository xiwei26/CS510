import sys
import cv2
import numpy as np
import mosse
import common

def check_overlap(obj, t_objs):
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
	#total_frame_num = 50
	### initialize MOG model ###
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	#fgbg = cv2.createBackgroundSubtractorMOG2()

	trackers = []
	tracking_objects = []

	for i in range(total_frame_num):
		#print i
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
			track_no = -1
			for j in range(len(trackers)):
				track_no = j
				trackers[j].update(gray_frame)	
				if trackers[j].good:
					x1,y1,x2,y2 = trackers[j].getRect()
					#trackers[j].draw_state(gray_frame)
					trackers[j].draw_state(frame, (track_no+1))
					tracking_objects[j].append((i,(x1,y1,x2,y2),trackers[j].psr))
					#print "%d,%d,%d,%d,%d,%d" % (j,i,x1,y1,x2,y2)

			for obj in moving_objects:
				track_no += 1
				if not check_overlap(obj,tracking_objects):
					new_tracker = mosse.MOSSE(gray_frame, obj)
					#new_tracker.draw_state(gray_frame)
					new_tracker.draw_state(frame, (track_no+1))
					trackers.append(new_tracker)
					tracking_objects.append([(i, obj, new_tracker.psr)])
					

	
		filename = './result/' + str(i) + '.png'
		#cv2.imwrite(filename,gray_frame)
		cv2.imwrite(filename, frame)
	#print(tracking_objects[0][0])
	write_tracks(tracking_objects)
