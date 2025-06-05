#!/bin/bash
echo "Executing Unet_CD model with LD_PRELOAD..."
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 /home/jetson/Desktop/CWFID/main.py "$1" "$2"
echo "Task completed."