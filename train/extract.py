# ------------------------------------------------------------------------------
# File Name         : extract.py
# Author            : Alberto Sánchez Sánchez
# Creation Date     : 2025-04-07
# Description       : Load class to read image and mask pairs from specific paths
#                     and generate a DataFrame for AI training purposes.
#
# Copyright         : © 2025 Alberto & DTE UPM department. All rights reserved.
# License           : This code is private and may not be distributed without 
#                     explicit authorization from the author and the department.
#                     For academic or research use, please contact the author
#                     to request permission.
#Email              : vicente.hernandez@upm.es / alberto.sanchez33@alumnos.upm.es
# ------------------------------------------------------------------------------

import cv2
import os
import pandas as pd 

class Load:
    "Represents a class asociated to the load of pair original and mask images to train later an IA algorithm"
    def __init__(self,image_path, mask_path):
        self.image_path = image_path #Path to the image
        self.mask_path = mask_path #Path to the mask

    def load_images(self):
        self.original_image = cv2.imread(self.image_path) #Loads original image
        self.mask_image = cv2.imread(self.mask_path) #Loads mask image

        if self.original_image is None:
            raise FileNotFoundError (f"No se puedo leer la imagen: {self.image_path}")
        if self.mask_image is None:
            raise FileNotFoundError (f"No se pudo leer la máscara: {self.mask_path}")

    """""
    def show():
        if self.original_image is not None and self.mask_image is not None:
            self.load_images()

        orginal_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2RGB)
    """""

def get_dataframe(images_path="dataset-1.0/images", mask_path="dataset-1.0/annotations"):
    images = [f for f in os.listdir(images_path) if f.endswith(".png")] #Extracts the .png files
    masks = [f for f in os.listdir(mask_path) if f.endswith(".png")] #Extracts the .png files
   
    images.sort()
    masks.sort()

    data = [] #List to store the data in a DataFrame
    for image, mask in zip(images, masks):
        data.append({"img_path": os.path.join(images_path, image), 
                     "mask_path":os.path.join(mask_path, mask)})
        
    return pd.DataFrame(data) #Convert data to DataFrame

if __name__ == "__main__":
    df = get_dataframe()
    print(df)



        