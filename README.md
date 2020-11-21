<h1 align="center">
	Object Tracking
</h1>

<div align="center">
	<!-- <img src="/media/manoj/Manoj_drive/Computer_vision/object-tracking/Images/person.gif" width="30%" height="30%"> -->
	<img src="https://github.com/Manojkl/object-tracking/blob/main/Images/person.gif" width="30%" height="30%"> 

</div>
Object tracking is the process of localizing the detected semantic object and track the object in real time. Objetcs are typically human, animal, cars , etc 

### Dependencies:
 - Python (3.7.4)
 - opencv_python (4.2.0.34)
 - imutils (0.5.3)
 - scipy (1.4.1)
 - numpy (1.19.0)

 ### Usage 
- Clone the repository to the local machine using ```git clone.```
- Make sure the requirment packages are installed 
```sh
pip install -r requirements.txt
```
- Change the directionry location to ```..\code\```
- The caffe model is pre-trained to detect human face.
- Run the below code to detect and track the face in real time in the terminal.
``` sh
python track_object_real_time.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
```
- Run the below code to detect and track the face from video file in the terminal.
``` sh
python track_object_from_video_file.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel --video ../object-tracking/code/test.mp4
```


