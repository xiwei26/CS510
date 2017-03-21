trackers = []
tracking_objects = []
for each frame:
	for i in range(len(trackers)):
		t.update(frame)
		tracking_objects[i].append((object_x, object_y, object_width, object_height, psr))
	moving_objects = [find_moving_objects()] # mix of gaussian to identify moving objects, eliminating small objects
	new_objects = []
	for obj in moving_objects:
		if obj not in tracking_objects:   #if identified object was not in the tracking results, indicating a new object
			new_objects.append()
	for new_obj in new_objects:  
		new_track = MOSSE(frame, new_obj)  # create new tracker for each new object
		trackers.append(new_track)
		tracking_objects.append([(object_x, object_y, object_width, object_height, psr)])  #for each new object, record its position

f = open('result.csv', 'w')
f.write(tracking_objects)