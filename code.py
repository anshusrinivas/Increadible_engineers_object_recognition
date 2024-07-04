from imutils.video import VideoStream
import numpy as np
import sys
import argparse
import imutils
import time
import cv2
from urllib.request import urlopen


host_url = 'http://192.168.0.101:8080/'
url = host_url + 'shot.jpg'

arguement_phrase = argparse.ArgumentParser()
arguement_phrase.add_argument("--prototxt", required=True,
	help="path to 	")
arguement_phrase.add_argument("--model", required=True,
	help="path to Caffe pre-trained model")
arguement_phrase.add_argument("--source", required=True, 
	help="Source of video stream (webcam/host)")
arguement_phrase.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
arguement = vars(arguement_phrase.parse_args())


CLASSES = ["background", 
		   "aeroplane", 
		   "bicycle", 
		   "bird", 
		   "boat",
			"bottle", 
			"bus", 
			"car", 
			"cat", 
			"chair",
			"cow", 
			"dining table",
			"dog",
			"horse", 
			"motor bike", 
			"person", 
			"plant", 
			"sheep",
			"sofa", 
			"train", 
			"tv or monitor"] # list of items model is trained on

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))# choose random colour for box


print("[INFORMATION] loading model...")
model = cv2.dnn.readNetFromCaffe(arguement["prototxt"], arguement["model"])
print("[INFORMATION] Model initialisation successful :)")

print("[INFORMATION] starting video stream...")

if arguement["source"] == "webcam":
	video_src = cv2.VideoCapture(0)

time.sleep(2.0) # delay 2 seconds to initialize the camera sensor

print("[INFORMAION] Camera initialised :)")


detected_objects = []

while True:
	if arguement["source"] == "webcam":
		ret, frame = video_src.read()
	else:
		imgResp=urlopen(url)
		imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
		frame=cv2.imdecode(imgNp,-1)
	
	frame = imutils.resize(frame, width=800)
		
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	
	model.setInput(blob)#give input to the model
	detections = model.forward()
    
	for i in np.arange(0, detections.shape[2]):# shape is extraceted as an array of numbers
		
		confidence = detections[0, 0, i, 2]
		if confidence > arguement["confidence"]:
			
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			detected_objects.append(label)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			
	
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	
	if key == ord("q"):# if q is pressed the exit the viewer
		break

cv2.destroyAllWindows()
