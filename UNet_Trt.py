# ------------------------------------------------------------------------------
# File Name         : U-NetTrt.py
# Author            : Alberto Sánchez Sánchez
# Creation Date     : 2025-04-08
# Description       : Load .h5 trained model and convert to optimized TensorRT   
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

IMG_SIZE = 256
N_CLASSES = 3 #Red, green and black(for background)

# -----------------------------------------------------------------
# FUNCTION:load_trt_model()-> loads trt model to later inference
#------------------------------------------------------------------
def load_trt_model(model_path = "trt_model"):
    return tf.saved_model.load(model_path)


# -----------------------------------------------------------------
# FUNCTION:predict_image()-> makes prediction of UnetTrt model
#------------------------------------------------------------------
def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_tensor = np.expand_dims(img_resized / 255.0, axis = 0).astype(np.float32)

    inference = model.signatures["serving_default"] #Function that makes the prediction when loading model
    pred = inference(tf.constant(input_tensor))['output_0'] #Prediction
    pred_mask = np.argmax(pred.numpy()[0], axis = -1).astype(np.uint8)

    colors = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
    }

    color_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for val, color in colors.items():
        color_mask[pred_mask == val] = color

    pred_combined = cv2.addWeighted(img_resized, 0.4, color_mask, 0.6)
    cv2.imshow("UNet_Prediction",pred_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
