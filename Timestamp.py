import cv2
import numpy as np
import pandas as pd
import time
import csv
import argparse
# The path to the Caffe prototxt file.
prototxt = './weights/deploy.prototxt'


# The path to the pretrained Caffe model.
model = './weights/weights.caffemodel'


# for counting how many images detected in video
counter = 0
confidence_thr = 0.5
cap = cv2.VideoCapture('./video/sample.mp4')
net = cv2.dnn.readNetFromCaffe(prototxt, model)
ip_time = time.strptime(input('Specify start time [HH.MM.SS]: '), "%H.%M.%S")
stop_time = time.strptime(input('Specify end time [HH.MM.SS]: '), "%H.%M.%S")

seconds = (ip_time.tm_hour*60*60*1000 + ip_time.tm_min*60*1000 + ip_time.tm_sec*1000)/1000
stop_seconds = (stop_time.tm_hour*60*60*1000 + stop_time.tm_min*60*1000 + stop_time.tm_sec*1000)/1000

cap.set(0,seconds*1000)
print('playing from ' + str(int(seconds)) + ' seconds')
print('till ' + str(int(stop_seconds)) + ' seconds')
print('press q to abort')
timestamp = 0

faces = [];  #list containing the times at which faces occur in the stream

st_time = time.time() 
while(timestamp<=stop_seconds):

    ret, frame = cap.read()
    timestamp = seconds + time.time() - st_time
    
    
    frame = cv2.putText(frame, "{0:.0f}".format(timestamp), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)

    if str(type(frame)) == "<class 'NoneType'>":
        break
    


    #kernel = np.array([[0,-1,0], [-1,8,-1], [0,-1,0]])
    #frame = cv2.filter2D(frame, -1, kernel)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
    net.setInput(blob)
    detections = net.forward()


    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        
        faces.append(int(timestamp))
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        coordinate = (startX, startY, endX, endY) = box.astype("int")
        detected_face = frame[startY:endY, startX:endX]
        name_for_face = './Cropped_Faces/face_at_'+str(int(timestamp))+'.jpg'
        counter += 1
        
        cv2.imwrite(name_for_face, detected_face)
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break   
cap.release()
cv2.destroyAllWindows()    


faces_unique = [];            #list containing the unique times (in seconds) at which faces occured  

for i in range(1,len(faces)):
    if(faces[i] != faces[i-1]):
        faces_unique.append(faces[i])
        
