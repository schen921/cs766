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
import copy
import math

class RSM:
    
    def __init__(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        # Create a config and configure the pipeline to stream
        config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        

        # For removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 2 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale


    def video(self):
        
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        try:
            while True:
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
        
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
        
                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame() 
                color_frame = aligned_frames.get_color_frame()
                
                self.depth_frame = depth_frame
        
                # Validate that both frames are valid
                if not depth_frame or not color_frame:
                    continue
        
                color_image = np.asanyarray(color_frame.get_data())
                self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                 
                depth_color_frame = rs.colorizer().colorize(depth_frame)
                 
                depth_color_image = np.asanyarray(depth_color_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                color_cvt = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                self.show(color_cvt)
                # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                # cv2.imshow('Align Example', color_image)
                # key = cv2.waitKey(1)
                 
        finally:
                self.pipeline.stop()
                
    def show(self,img):
        
        self.img_origin = img
        self.img_copy = copy.copy(self.img_origin)
        cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
        
        cv2.setMouseCallback("Color Stream", self.draw)
        while True:
            cv2.imshow("Color Stream", self.img_origin)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            
            
    def draw(self, event,x,y,flags,params):
        img = copy.copy(self.img_copy)
        #print event,x,y,flags,params
        if(event==1):
            self.ix = x
            self.iy = y
        elif event == 4:
            img = self.img_copy
            self.img_work(img, x,y)
        elif event == 2:
            self.img_copy = copy.copy(self.img_origin)
        elif(flags==1):
            self.img_work(img,x,y)
            cv2.imshow("Color Stream", img)
            
            
    def img_work(self, img,x,y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0, 0, 0)
        lineType = 2

        ans = self.calculate_distance(x, y)
        cv2.line(img, pt1=(self.ix, self.iy), pt2=(x, y), color=(255, 255, 255), thickness=3)
        cv2.rectangle(img, (self.ix, self.iy), (self.ix + 80, self.iy - 20), (255, 255, 255), -1)
        cv2.putText(img, '{0:.5}'.format(ans), (self.ix, self.iy), font, fontScale, fontColor,
                    lineType)
        
        
    def calculate_distance(self,x,y):
        color_intrin = self.color_intrin
        ix,iy = self.ix, self.iy
        udist = self.depth_frame.get_distance(ix,iy)
        vdist = self.depth_frame.get_distance(x, y)
        #print udist,vdist

        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [ix, iy], udist)
        point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], vdist)
        #print str(point1)+str(point2)

        dist = math.sqrt(
            math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(
                point1[2] - point2[2], 2))
        
        print('distance: '+ str(dist))
        return dist

if __name__ == '__main__':
    RSM().video()        