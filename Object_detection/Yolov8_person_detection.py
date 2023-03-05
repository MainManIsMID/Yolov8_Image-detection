import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
import glob
from PIL import Image

model = YOLO('yolov8n.pt')
model.train(data = 'coco128.yaml' ,epochs=5)
model.val()

folder_path = "#path to the folder/file" 

image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]


results = []
for image_file in image_files:
    with Image.open(image_file) as img:
        print(os.path.basename(image_file))
    results.append(model(image_file))
    res = model(image_file).copy()
    has_person = False
    for r in results[-1]:
        boxes = r.boxes
        
        for index in range(len(boxes.conf)):
            if boxes.conf[index] >= 0.5 and boxes.cls[index] == 0:
               has_person = True
            
        
        
    
        
        if has_person:
            res_plotted = res[0].plot()

       
            resized_img = cv2.resize(res_plotted, (800, 600))
            cv2.imshow("Image_detections", resized_img)
        
    
    ESC = cv2.waitKey(0)
    
    if ESC == 27:
        cv2.destroyAllWindows()
       


    
