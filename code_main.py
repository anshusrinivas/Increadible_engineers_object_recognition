from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from urllib.request import urlopen
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import pyttsx3 as py

en = py.init()
en.setProperty('rate', 125)
en.setProperty('volume', 150)

host_url = 'http://192.168.0.101:8080/'
url = host_url + 'shot.jpg'

protoxt = "MobileNetSSD_deploy.prototxt.txt"
model_obj_path = "MobileNetSSD_deploy.caffemodel"
source = "webcam"
confidence_threshold = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
           "dining table", "dog", "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "tvmonitor"]
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFORMATION] Loading models...")

# Load object detection model
model_obj = cv2.dnn.readNetFromCaffe(protoxt, model_obj_path)

# Load emotion recognition model
model_emo = Sequential()
model_emo.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model_emo.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_emo.add(MaxPooling2D(pool_size=(2, 2)))
model_emo.add(Dropout(0.25))
model_emo.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_emo.add(MaxPooling2D(pool_size=(2, 2)))
model_emo.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_emo.add(MaxPooling2D(pool_size=(2, 2)))
model_emo.add(Dropout(0.25))
model_emo.add(Flatten())
model_emo.add(Dense(1024, activation='relu'))
model_emo.add(Dropout(0.5))
model_emo.add(Dense(7, activation='softmax'))
model_emo.load_weights('model.h5')

print("[INFORMATION] Models initialized successfully :)")

print("[INFORMATION] Starting video stream...")

if source == "webcam":
    video_src = cv2.VideoCapture(0)
else:
    video_src = None

time.sleep(2.0)
cv2.ocl.setUseOpenCL(False)

print("[INFORMATION] Camera initialized :)")

while True:
    if source == "webcam":
        ret, frame = video_src.read()
    else:
        imgResp = urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    model_obj.setInput(blob)
    detections = model_obj.forward()

    speech = ""
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            if CLASSES[idx] == 'person':
                facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_image = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    pred = model_emo.predict(cropped_image)
                    maxind = int(np.argmax(pred))
                    cv2.putText(frame, emotion_dict[maxind], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (255, 255, 255), 2, cv2.LINE_AA)
                    speech += f'There is a person in front of you. They look {emotion_dict[maxind]}. '
            else:
                speech += f'There is a {CLASSES[idx]} in front of you. '

    if speech:
        en.say(speech)
        en.runAndWait()
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
if video_src:
    video_src.release()
