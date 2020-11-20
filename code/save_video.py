from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# vs = cv2.VideoCapture('test.mp4')
# time.sleep(2.0)

vs = cv2.VideoCapture(0)

_, frame = vs.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/manoj/HBRS/output.avi',fourcc, 20.0, (fwidth,fheight))

while(vs.isOpened()):
    ret, frame = vs.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
vs.release()
out.release()
cv2.destroyAllWindows()

