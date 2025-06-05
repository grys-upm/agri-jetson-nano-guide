# ------------------------------------------------------------------------------
# File Name         : extract.py
# Author            : Alberto Sánchez Sánchez
# Creation Date     : 2025-04-07
# Description       : Load statistics to Database for later consulting and 
#                     display
# Copyright         : © 2025 Alberto & DTE UPM department. All rights reserved.
# License           : This code is private and may not be distributed without 
#                     explicit authorization from the author and the department.
#                     For academic or research use, please contact the author
#                     to request permission.
#Email              : vicente.hernandez@upm.es / alberto.sanchez33@alumnos.upm.es
# ------------------------------------------------------------------------------

import pymysql

def connect_to_mysql(host, user, password, db):
    try:
        conn = pymysql.connect(
            host = host,
            user = user,
            password = password,
            database = db
        )
        return conn
    
    except pymysql.MySQLError as err:
        print(f"Error connecting to MySql {err}")
        return None
    
def insert_data(conn, stats):
    cursor = conn.cursor()

    sql = """
        INSERT INTO Carrot_Herbicide (id, timestamp, img_path, pred_path, green, red ,background)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            timestamp = VALUES(timestamp),
            img_path = VALUES(img_path),
            pred_path = VALUES(pred_path),
            background = VALUES(background),
            green = VALUES(green),
            red = VALUES(red)
        """
    values = (stats["id"], stats["timestamp"], stats["img_path"], stats["pred_path"], stats["green"], stats["red"], stats["background"])
    cursor.execute(sql, values)
    conn.commit()
    cursor.close()

    
