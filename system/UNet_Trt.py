# ------------------------------------------------------------------------------
# File Name         : U-NetTrt.py
# Author            : Alberto Sánchez Sánchez
# Creation Date     : 2025-04-08
# Description       : Load TensorRT model and make prediction and statistics   
# Copyright         : © 2025 Alberto & DTE UPM department. All rights reserved.
# License           : This code is private and may not be distributed without 
#                     explicit authorization from the author and the department.
#                     For academic or research use, please contact the author
#                     to request permission.
#Email              : vicente.hernandez@upm.es / alberto.sanchez33@alumnos.upm.es
# ------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import cv2
import os
import time
import re
from datetime import datetime, timedelta

IMG_SIZE = 256
N_CLASSES = 3 #Red, green and black(for background)

# -----------------------------------------------------------------
# FUNCTION:load_trt_model()-> loads trt model to later inference
#------------------------------------------------------------------
def load_trt_model(model_path = "saved_model"):
    return tf.saved_model.load(model_path)


# --------------------------------------------------------------------------------------
# FUNCTION:predict_image()-> makes prediction of UnetTrt model and creates tabular stats
#---------------------------------------------------------------------------------------
def predict_image(model, image_path, output_dir="predictions", time_step=0):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_tensor = np.expand_dims(img_resized / 255.0, axis = 0).astype(np.float32)

    inference = model.signatures["serving_default"] #Function that makes the prediction when loading model

    start_time = time.time()
    pred = inference(tf.constant(input_tensor))['softmax'] #Prediction
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"⏱️ Inference time: {elapsed_time:.4f} segundos")
    
    pred_mask = np.argmax(pred.numpy()[0], axis = -1).astype(np.uint8)

    colors = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
    }

    color_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for val, color in colors.items():
        color_mask[pred_mask == val] = color

    pred_combined = cv2.addWeighted(img_resized, 0.6, color_mask, 0.4, 0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"pred_{filename}_Jetson_t+{time_step}.png")
    cv2.imwrite(output_path, pred_combined)
    print(f"Prediction saved at: {output_path}")

    #Calculate stats and stores it in database
    
    filename_no_ext = os.path.splitext(os.path.basename(output_path))[0]  # Ej: pred_001_image_Jetson_t+5
    match_plant = re.search(r"pred_(\d+)_", filename_no_ext)
    plant_num = match_plant.group(1) if match_plant else "000"

    match_time = re.search(r"t\+(\d+)", filename_no_ext)
    instance = match_time.group(1) if match_time else "0"

    primary_key = f"{plant_num}_{instance}"

    now = datetime.now()
    base_time = now.replace(hour=10, minute=0, second=0)
    time_offset = int(instance)
    timestamp = base_time + timedelta(days=15 * time_offset)

    num_pixels = pred_mask.size
    stats = {
        "id" : primary_key,
        "timestamp" : timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "img_path": os.path.abspath(image_path), #Absolute path to find the original image
        "pred_path" : os.path.abspath(output_path), #Absolute path for the prediction image
        "green" : round(np.sum(pred_mask == 1) / num_pixels * 100, 2),
        "red" : round(np.sum(pred_mask == 2) / num_pixels * 100, 2),
        "background": round(np.sum(pred_mask == 0) / num_pixels * 100, 2)
    }
 

    return stats
 
