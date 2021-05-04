# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:44:17 2021

@author: sinas
"""

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import matplotlib.pyplot as plt
import time

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 2 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
       
        frameset = pipeline.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
       
        color = np.asanyarray(color_frame.get_data())
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = [12, 6]
        # plt.imshow(color)
        
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        # plt.imshow(colorized_depth)
        
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)
        
        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        
        # Show the two frames together:
        images = np.hstack((color, colorized_depth))
        # plt.imshow(images)
        
        
        # Standard OpenCV boilerplate for running the net:
        height, width = color.shape[:2]
        expected = 300
        aspect = width / height
        resized_image = cv2.resize(color, (round(expected * aspect), expected))
        crop_start = round(expected * (aspect - 1) / 2)
        crop_img = resized_image[0:expected, crop_start:crop_start+expected]
        
        net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ln = net.getLayerNames()
        
        blob = cv2.dnn.blobFromImage(crop_img, 1/255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]
        
        cv2.imshow('blob', r)
        text = f'Blob shape={blob.shape}'
        cv2.waitKey(1)
        
        net.setInput(blob)
        t0 = time.time()
        outputs = net.forward(ln)
        t = time.time()
        
        #cv2.displayOverlay('window', f'forward propagation time={t-t0}')
        cv2.imshow('window',  crop_img)
        cv2.waitKey(0)
        
        net.setInput(blob)
        outputs = net.forward(ln)
        
        cv2.destroyAllWindows()
        # inScaleFactor = 0.007843
        # meanVal       = 127.53
        # classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
        #               "bottle", "bus", "car", "cat", "chair",
        #               "cow", "diningtable", "dog", "horse",
        #               "motorbike", "person", "pottedplant",
        #               "sheep", "sofa", "train", "tvmonitor")
        
        # blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
        # #net.setInput(blob, "data")
        # #detections = net.forward("detection_out")
        
        # label = detections[0,0,0,1]
        # conf  = detections[0,0,0,2]
        # xmin  = detections[0,0,0,3]
        # ymin  = detections[0,0,0,4]
        # xmax  = detections[0,0,0,5]
        # ymax  = detections[0,0,0,6]
        
        # className = classNames[int(label)]
        
        # cv2.rectangle(crop_img, (int(xmin * expected), int(ymin * expected)), 
        #              (int(xmax * expected), int(ymax * expected)), (255, 255, 255), 2)
        # cv2.putText(crop_img, className, 
        #             (int(xmin * expected), int(ymin * expected) - 5),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
        
        # plt.imshow(crop_img)        
                       
        # # Get frameset of color and depth
        # frames = pipeline.wait_for_frames()
        # # frames.get_depth_frame() is a 640x360 depth image

        # # Align the depth frame to color frame
        # aligned_frames = align.process(frames)

        # # Get aligned frames
        # aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        # color_frame = aligned_frames.get_color_frame()

        # # Validate that both frames are valid
        # if not aligned_depth_frame or not color_frame:
        #     continue

        # depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        # # Remove background - Set pixels further than clipping_distance to grey
        # grey_color = 153
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # # Render images:
        # #   depth align to color on left
        # #   depth on right
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # #images = np.hstack((bg_removed, depth_colormap))

        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow('Align Example', bg_removed)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
