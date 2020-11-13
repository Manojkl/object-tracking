## Centroid tracking with OpenCV

* The centroid tracking algorithm depends on the euclidean distance between

   a) Existing object centroid: The objects the centroid tracker has already seen before.  
  
   b) The new object centroid found between the subsequent frames of a video.

## The centroid tracking algorithm

Step1: Find the bounding box coordinates and compute the centroid
* Assumptions: The bounding box coordinates (x,y) are provided to the algorithm.
* The bounding box can be extracted for each frame using different object detector algorithms such as color thresholding + contour extraction, Haar cascades, HOG + Linear SVM, SSDs, Faster R-CNNs, etc.
* The centroid (x,y) of the bounding box is computed.

<p align="center"><img src="https://github.com/Manojkl/object-tracking/blob/main/Documents/simple_object_tracking_step1.png" width="400" height="250" />

* A unique id is assigned to the detected bounding boxes.

Step2: Euclidean distance between the new and existing bounding boxes are computed

* For every frame in the video step1 is computed, however we need to associate the new object centroid to the old object centroid. To do this the euclidean distance between the new centroid and old centroid is calculated.
* In the image below we can see we have detected three objects in our image. The two pairs that are close together are two existing objects.

<p align="center"><img src="https://github.com/Manojkl/object-tracking/blob/main/Documents/simple_object_tracking_step2.png" width="400" height="250" />

* Euclidean distance between each pair of original centroid and new centroids are computed.

Step3: Existing object (x,y) cooridnate is updated

* One of the primary assumption of centroid algorithm is that the a given object will potentially move in between the frames, but the distance between the centroid frames $F_t$ and $F_{t+1}$ is smaller than all other distance between objects.

* If we are able to find the centroids with the minimum distance between them then we can do the tracking of the object.

* What if we have some left out centroid? What we do with it?

<p align="center"><img src="https://github.com/Manojkl/object-tracking/blob/main/Documents/simple_object_tracking_step3.png" width="400" height="250" />

* We assign them as the new objet.

Step4: Register new objects

* If the detected objects are more than the existing objects then new object is registered. It means to say we are adding the new object to our list of tracked objects. By giving new ID and storing the centroid of the bounding box coordinates.

<p align="center"><img src="https://github.com/Manojkl/object-tracking/blob/main/Documents/simple_object_tracking_step4.png" width="400" height="250" />

* Go to step 2 and repeat the procedure for every frame.

Step5: Deleting old objects

* When the object is lost after checking N frames then the old objects is deleted.





