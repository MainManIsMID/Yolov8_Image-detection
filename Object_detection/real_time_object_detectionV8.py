import torch
import cv2
from matplotlib import pyplot as plt
from numpy import ndarray as nd
from ultralytics import YOLO
import pandas as pd


model = YOLO('yolov8n.pt')

#imgs = ["https://daily.jstor.org/wp-content/uploads/2017/12/traffic_jam_1050x700.jpg"]


#results = model(imgs)

cap = cv2.VideoCapture(2) #this depends try 0,1 or 2 on device
while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)
    
    cv2.imshow('Img-Detection', nd.squeeze(results[0].plot()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
