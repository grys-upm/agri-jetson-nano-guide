# ------------------------------------------------------------------------------
# File Name         : extract.py
# Author            : Alberto Sánchez Sánchez
# Creation Date     : 2025-04-07
# Description       : Main agent to execute all the Carrot Detection System
# Copyright         : © 2025 Alberto & DTE UPM department. All rights reserved.
# License           : This code is private and may not be distributed without 
#                     explicit authorization from the author and the department.
#                     For academic or research use, please contact the author
#                     to request permission.
#Email              : vicente.hernandez@upm.es / alberto.sanchez33@alumnos.upm.es
# ------------------------------------------------------------------------------

from UNet_Trt import load_trt_model, predict_image
from Load_DB import connect_to_mysql, insert_data
import sys, os

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("You should proporcionate a correct image")
        sys.exit(1) #Advice something went wrong in the system

    if len(sys.argv) >= 3:
        time_step = int(sys.argv[2])
    else:
        time_step = 0

    image_path = sys.argv[1]
    model = load_trt_model("saved_model") 
    stats = predict_image(model, image_path, time_step = time_step)

    conn = connect_to_mysql(
        host = "localhost",
        user = os.environ.get("CD_USER"),
        password = os.environ.get("CD_PASSWORD"),
        db = os.environ.get("DB_NAME")
    )

    if conn:
        insert_data(conn, stats)
        conn.close()
        print("Data saved in Database")
    else:
        print("Error in the conection with database")


