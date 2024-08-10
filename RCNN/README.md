RCNN based object detection from images using Tensorflow, Keras, OpenCV

Steps:

1. Raccoon Dataset preparation
2. Build dataset using selective search
3. Use the dataset to train the pre-trained MobileNet (trained on Imagenet) to detect the object in the image. 
4. Apply Non Max Suppression to detect the region with highest probability of the detected object 

<div align="center">
	<!-- <img src="/media/manoj/Manoj_drive/Computer_vision/object-tracking/Images/person.gif" width="30%" height="30%"> -->
	<img src="https://github.com/Manojkl/object-tracking/blob/main/Images/raccoon.png" width="70%" height="70%">

</div>

References:

1. https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/ 
