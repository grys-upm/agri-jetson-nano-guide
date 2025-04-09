# ------------------------------------------------------------------------------
# File Name         : AI_algorithm.py
# Author            : Alberto Sánchez Sánchez
# Creation Date     : 2025-04-08
# Description       : AI algorithm used to do segmentation classification of good 
#                     carrots and brush. Decision making support.
#
# Copyright         : © 2025 Alberto & DTE UPM department. All rights reserved.
# License           : This code is private and may not be distributed without 
#                     explicit authorization from the author and the department.
#                     For academic or research use, please contact the author
#                     to request permission.
#Email              : vicente.hernandez@upm.es / alberto.sanchez33@alumnos.upm.es
# ------------------------------------------------------------------------------

from extract import get_dataframe

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import os

IMG_SIZE = 256
N_CLASSES = 3 #Red, green and black(for background)
BATCH_SIZE = 2
EPOCHS = 15

GREEN = [0,255,0]
RED = [0, 0, 255]

MODEL_PATH = "model/seg.model.h5"

# -----------------------------------------------------------------
# FUNCTION: convert_mask_to_class()-> converts RGB mask to an array
#------------------------------------------------------------------
def convert_mask_to_class(mask_rgb):
    h, w, _ = mask_rgb.shape
    mask_class = np.zeros((h,w), dtype=np.uint8)
    mask_class[np.all(mask_rgb == GREEN, axis=-1)] = 1
    mask_class[np.all(mask_rgb == RED, axis=-1)] = 2
    return mask_class

# ------------------------------------------------------------------
# FUNCTION: data_from_df-> load data from df
#------------------------------------------------------------------
def data_from_df(df):
    X, Y = [], []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img = cv2.imread(row['img_path'])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        mask_rgb = cv2.imread(row['mask_path'])
        mask_rgb = cv2.resize(mask_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_class = convert_mask_to_class(mask_rgb)

        X.append(img)
        Y.append(mask_class)

    X = np.array(X) / 255
    Y = np.expand_dims(np.array(Y), axis = -1)
    Y_cat = tf.keras.utils.to_categorical(Y, num_classes=N_CLASSES)

    #75% used for training, 20% for testing and 5% for validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.2, random_state=42)

    print("Índices únicos de X_val:", np.unique(X_val))
    print("¿Hay índices repetidos en X_val?", len(X_val) != len(np.unique(X_val))) 

    return X_train, X_test, X_val, Y_train, Y_test, Y_val
# --------------------------------------------------------------------
# FUNC: fine_tuning ()-> Train the DeepLabV3 with MobilenetV2 backbone
#                        with the data we are interested in
# --------------------------------------------------------------------
def train_model(X_train, Y_train, X_test, Y_test):
    print("Training DeepLabV3 model + MobileNetV2")

    adam_optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,    # Learning rate 
        beta_1=0.9,              # Momentum 
        epsilon=1e-8             # Epsilon
    )

    sm.set_framework('tf.keras')
    sm.framework()
    model = sm.Unet('mobilenetv2', classes=N_CLASSES, activation='softmax')

    model.compile(optimizer=adam_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, Y_test))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
    return model

# --------------------------------------------------
# FUNC: predict_and_show() -> do a visual prediction
# --------------------------------------------------
def predict_and_show(model, img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_tensor = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

    pred = model.predict(input_tensor)
    pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

    colors = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
    }

    color_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for val, color in colors.items():
        color_mask[pred_mask == val] = color

    combined = cv2.addWeighted(img_resized, 0.6, color_mask, 0.4, 0)
    cv2.imshow("Predicción", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------------------
# FUNC: get_image_from_val() -> Prints a validation image
# --------------------------------------------------------------
def get_image_from_val(df, val_index):
    val_img = df.iloc[val_index]['img_path']
    print(f"Image used in the validation: {val_img}")
    return val_img


if __name__ == "__main__":

    df = get_dataframe()

    X_train, X_test, X_val, Y_train, Y_test, Y_val = data_from_df(df)
    model = train_model(X_train, Y_train, X_test, Y_test)

    #Show which validation images are selected to be predicted
    val_images = df.iloc[X_val.flatten()].reset_index(drop=True)['img_path'].tolist()
    print("Validation images:")
    #for idx, val_img in enumerate(val_images):
    #    print(f"{idx}: {val_img}")

    #Introduce by keyboard which validation image index is going to be predicted
    try:
        selected_idx = int(input("Please enter the index of the image to be predicted: "))
        if selected_idx < 0 or selected_idx >= len(val_images):
            raise ValueError("Index not exits")
    except ValueError as e:
        print(f"Error: {e}. Please enter a valid index")
    else:
        # Obtener la imagen seleccionada
        val_img_path = val_images[selected_idx]
        print(f"Showing prediction from selected image: {val_img_path}")
    
    predict_and_show(model, val_img_path)