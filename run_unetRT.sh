#!/bin/bash
echo "Ejecutando modelo UNet con LD_PRELOAD..."
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 /home/jetson/Desktop/CWFID/UNet_Trt.py
echo "Script finalizado."