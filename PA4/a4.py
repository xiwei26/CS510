import sys
import numpy as np
import cv2
import os

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from vgg16 import vgg16


'''
1. resizeImg() takes the frame and the feature window, producing a 224*224 image; if the input feature window is a rectangle,
	this function only takes the center part of the image

2. Put a4.py in the vgg folder 
'''

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

def checkOverlap(x1,y1,r1,x2,y2,r2):
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
#total_frame_num = 50
#####processing video#####
coordinatesList = []
hessian = 5000
ksize = 15 ### parameter for gaussian blur
M_enlarge = np.array([0]) ### used for resizing attention window when frame size is shrinked
fold = 2 ### shrinking fold

#sum_sz = 0 ### for calculating average attention window size
#large_sz = 0  ### for calculating maximum attention window size

### vgg object, copied from vgg16.py ###
sess = tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(images, 'vgg16_weights.npz', sess)

#image_stack = None
for i in range (total_frame_num):
	ret, frame = cap.read()

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
	image_stack = [img_tmp] #if image_stack == None else np.append(image_stack, [img_tmp], axis = 0)
	cv2.imwrite('./result/' + str(i) + '.jpg', img_tmp)

	### Predict object class ###
	probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})
	preds = np.argmax(probs, axis=1)
	for index, p in enumerate(preds):
	    print("Prediction #%d: %s; Probability: %f"%(i, class_names[p], probs[index, p]))


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