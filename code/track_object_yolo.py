# Running the script
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# Import all the requirements

from trackcentroid import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Build the argument parser
ap = argparse.ArgumentParser()
#  The path to the Caffe “deploy” prototxt.
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
# Use the trained model present in the opencv
# https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained Caffe model")
# Probability threshold to filter weak detections. A default value of 0.5 is sufficient.
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

    # if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

    # construct a blob from the frame, pass it through the network,
	# obtain output predictions, and initialize the list of
	# bounding box rectangles
    # mean -scalar with mean values which are subtracted from channels. 
	# Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
	# Blob from image is used as preprocessing before processing the data.
	# Blob from image perform mean subtraction, scaling and optional channel swapping
	# Mean subtraction is used for help us to combat illumination changes problem 
	# The three values mentioned in tuple are the mean of red,blue and green. For Imagenet dataset the mean values are R=103.93, G=116.77, and B=123.68.
	# https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

    # loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 255), 1)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
		cv2.circle(frame, (centroid[0], centroid[1]), 2, (255, 0, 0), -1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()
