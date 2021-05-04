
#####################################################
##              Align Depth to Color               ##
#####################################################
# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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
iter = 0
# Streaming loop
try:
    while True:
        iter = iter + 1
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)


        #convert color image to gray for circle detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #apply median blue
        gray_med = cv2.medianBlur(gray, 5)
        
        rows = gray.shape[0]
        #detect circles
        circles = cv2.HoughCircles(gray_med,cv2.HOUGH_GRADIENT, 1, 1,
                                    param1=30, param2=45,
                                    minRadius=1, maxRadius=35)
        
        edges = cv2.Canny(gray,25,40)
        mask = (~edges)
        
        
        
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # first circle
            j = circles[0, :]
            circle_cen = [j[0][0],j[0][1]]
            rad = j[0][2]
            
            x_left = int(circle_cen[0]-rad)
            x_left_l = int(circle_cen[0]-3*rad)
            y_up = int(circle_cen[1]-3*rad)
            y_down = int(circle_cen[1]+3*rad)
            
            
            dists_l = []
            for x in [x_left_l,x_left]:
                for y in [y_up,y_down]:
                   dist =  aligned_depth_frame.get_distance(x,y)
                   if dist > 0:
                       dists_l.append(dist)
            
            dists_left_mean = np.mean(dists_l)
            print("dist left:  " + str(dists_left_mean ))
                
       

        
        #draw the cicles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(mask, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
               # cv2.circle(bg_removed, center, radius, (255, 0, 255), 3)
                cv2.circle(mask, center, radius, (255, 0, 255), 3)
                
        cv2.imshow("detected circles", bg_removed)
        cv2.waitKey(1)
        
        
        # # Saving images each 50 iteration
        # if iter % 50 == 0:        
        #   current_dir = os.getcwd()
        #   save_dir = "images"
        #   path = os.path.join(current_dir, save_dir, 'image_cd' + str(iter) + '.jpg')    
        #   cv2.imwrite(path, bg_removed)
        
        
        
        ## Render images:
        ##   depth align to color on left
        ##   depth on right
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((bg_removed, depth_colormap))

        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow('Align Example', bg_removed)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
