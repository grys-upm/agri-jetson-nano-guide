# ------------------------------------------------------------------------------
# File Name         : Trt_model.py
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
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import numpy as np
import cv2

model = tf.keras.models.load_model("model/seg.model.h5", compile=False) #Load saved model 
tf.saved_model.save(model, "saved_model") #Save the optimized model for TensorRT

params = trt.TrtConversionParams(
    precision_mode = trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes= 1 << 28
)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir = "Unet_trt_model",
    conversion_params = params
)

converter.convert()
converter.save("Unet_trt.model")










