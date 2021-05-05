import pyrealsense2 as rs
import numpy as np
import cv2
import os

# For the Mask-RCNN model

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from numpy import expand_dims

from mrcnn.model import mold_image



# define the prediction configuration
class PredictionConfig(Config):

    # define the name of the configuration
    NAME = "part_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


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

config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

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

# Load the model for inference
# create config
cfg = PredictionConfig()
# define the model
model_dir = os.path.join(os.getcwd(), "models")
model = MaskRCNN(mode='inference', model_dir=model_dir, config=cfg)

# load model weights
weights_dir = os.path.join(os.getcwd(), "models",
                           'mask_rcnn_part_cfg_0010.h5')
model.load_weights(weights_dir, by_name=True)


# Streaming loop
try:
    while True:
        iter = iter + 1

        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame()

        color_frame = aligned_frames.get_color_frame()

        # Get just the color frame
        color_image = np.asanyarray(color_frame.get_data())

        ###################################################
        # MaskRCNN Prediction
        ###################################################

        # Predict for each frame -> One by one
        # Convert pixel values
        scaled_color_img = mold_image(color_image, cfg)
        # Convert image to one sample
        sample = expand_dims(scaled_color_img, 0)
        # Make the prediction
        yhat = model.detect(sample, verbose=0)

        # Get the box parameters
        try:
            y1, x1, y2, x2 = yhat[0]["rois"][0]
        except:
            continue
        #width, height = x2 - x1, y2 - y1
        #rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        rec_color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)


        cv2.imshow("RGB8 Image", rec_color_image)
        cv2.waitKey(1)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break



finally:
    pipeline.stop()
