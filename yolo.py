import numpy as np
import argparse
import imutils
import time
import cv2

import os
import PIL
from tkinter import *
from timeit import default_timer as timer
import math 

font = cv2.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (0, 0, 255)
line_type = cv2.LINE_AA
IMAGE_SIZE = 224



def work(path):
    frame=cv2.imread(path)
    labelsPath = os.path.sep.join(["allmodel", "labels.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    detections=["person"]
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")

    weightsPath = os.path.sep.join(["allmodel", "yolov4.weights"])
    configPath = os.path.sep.join(["allmodel", "yolov4.cfg"])

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    frame1=frame
    

    
    (H, W) = frame1.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame1, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    boxes = []
    confidences = []
    classIDs = []
    # frame1=cv2.resize(frame,(500,500))
    
    

    for output in layerOutputs:
            for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    
                    if confidence > 0.2:
                            
                            box = detection[0:4] * np.array([W, H, W, H])
                            
                            (centerX, centerY, width, height) = box.astype("int")

                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.2,0.2)
    flag=0
    if len(idxs) > 0:
            for i in idxs.flatten():
                if LABELS[classIDs[i]] in detections:
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        if LABELS[classIDs[i]] in ['person']:
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])
                                cv2.rectangle(frame1, (x, y), (x+ w, y +h), (0,255,0), 2)

    cv2.imwrite('res.jpg',frame1)                         

                                                    
                                
                            
           
           
                    


                

    
 

##showface()
