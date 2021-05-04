
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math

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
clipping_distance_in_meters = 1 #1 meter
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


        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
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
                                    param1=25, param2=40,
                                    minRadius=1, maxRadius=30)
        
        edges = cv2.Canny(gray,35,40)
        mask = (~edges)
        
        filter_mask = mask/np.max(mask);
        
        filter_depth = np.multiply(filter_mask,depth_image)
        
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            
            circle_cen = [np.mean(np.round(circles[0,:,0])),np.mean(np.round(circles[0,:,1]))]
            rad = np.mean(np.round(circles[0,:,2]))

            
            # LEFT BOUNDARY OF THE CIRCLE
            x_l = int(circle_cen[0]-rad)
            x_l_l = int(circle_cen[0]-2*rad)
            y_l_up = int(circle_cen[1]-1*rad)
            y_l_down = int(circle_cen[1]+1*rad)
            
  
            # Get the left boundary's distance
            dists_l = []
            for x in range(x_l_l,x_l,1):
                for y in range(y_l_up,y_l_down,1):
                   dist =  aligned_depth_frame.get_distance(x,y)
                   if dist > 0:
                       dists_l.append(dist)
                       #print("Dist value at"+ str(x)+", "+str(y)+"  :"+str(dist))
            dists_l_mean = np.mean(dists_l)
            #print("dist left:  " + str(dists_l_mean ))
            
            # Right BOUNDARY OF THE CIRCLE
            x_r = int(circle_cen[0]+rad)
            x_r_r = int(circle_cen[0]+2*rad)
            y_r_up = int(circle_cen[1]-1*rad)
            y_r_down = int(circle_cen[1]+1*rad)
 

            # Get the right boundary's distance
            dists_r = []
            for x in range(x_r,x_r_r,1):
                for y in range(y_r_up,y_r_down,1):
                   dist =  aligned_depth_frame.get_distance(x,y)
                   if dist > 0:
                       dists_r.append(dist)
                       #print("Dist value at"+ str(x)+", "+str(y)+"  :"+str(dist))
            dists_r_mean = np.mean(dists_r)            
            #print("dist right:  " + str(dists_r_mean)) 
            
            
            avg_rad = []
            if abs(dists_r_mean-dists_l_mean)<0.005:
                for alpha in np.arange(0,math.pi,0.3):
                    
                    p1_x = int(circle_cen[0] + rad * math.cos(alpha))
                    p1_y = int(circle_cen[1] + rad * math.sin(alpha))
                    
                    p2_x = int(circle_cen[0] + rad * math.cos(alpha+math.pi))
                    p2_y = int(circle_cen[1] + rad * math.sin(alpha+math.pi))          
                
                    p1_z =  aligned_depth_frame.get_distance(p1_x,p1_y)
                    p2_z =  aligned_depth_frame.get_distance(p2_x,p2_y)
                
                    
                    if not (p2_z == 0) and not (p1_z == 0):
                        
                        point1 =  rs.rs2_deproject_pixel_to_point(color_intrin, [p1_x, p1_y], (p1_z))
                        point2 =  rs.rs2_deproject_pixel_to_point(color_intrin, [p2_x, p2_y],(p2_z))
                        dist = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(point1[2] - point2[2], 2))
                        if not np.isnan(dist):
                            avg_rad.append(dist)
                    
                    
                    
                    mean_avg_rad = np.mean(avg_rad)
                    if not np.isnan(mean_avg_rad):
                        print("Diameter : " + str(mean_avg_rad*1000)+" mm")
            
        #draw the cicles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(bg_removed, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
               
                cv2.circle(bg_removed, center, radius, (255, 0, 255), 3)
                
                #draw the boundary on left side of the circle
                cv2.rectangle(bg_removed,(x_l,y_l_up),(x_l_l,y_l_down),(0, 255, 255), 1)
                cv2.rectangle(bg_removed,(x_r,y_r_up),(x_r_r,y_r_down),(0, 255, 255), 1)
                
        cv2.imshow("detected circles",bg_removed)
        cv2.imshow("mask",mask)
        cv2.waitKey(1)
        
        
        # # Saving images each 50 iteration
        # if iter % 50 == 0:        
        #   current_dir = os.getcwd()
        #   save_dir = "images"
        #   path = os.path.join(current_dir, save_dir, 'image_cd' + str(iter) + '.jpg')    
        #   cv2.imwrite(path, bg_removed)
        
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
