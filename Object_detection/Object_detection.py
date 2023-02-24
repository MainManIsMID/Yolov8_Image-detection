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

folder_path = "#path to the folder/file"

image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

results = []
for image_file in image_files:
    with Image.open(image_file) as img:
        print(os.path.basename(image_file))
    results.append(model(image_file))
    res = model(image_file,conf=0.5)


    res_plotted = res[0].plot()

        
    resized_img = cv2.resize(res_plotted, (800, 600))
    cv2.imshow("Image_detections", resized_img)
    
    ESC = cv2.waitKey(0)
    if ESC == 27:
        cv2.destroyAllWindows()
