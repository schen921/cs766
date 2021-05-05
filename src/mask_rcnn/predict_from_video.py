import numpy as np
import cv2
import os

# For the Mask-RCNN model

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from numpy import expand_dims

from mrcnn.model import mold_image


# Open up the Video file
video = cv2.VideoCapture("raw_video.avi")
# Read the file
status, color_image = video.read()
count = 0

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
pred_video_writer = cv2.VideoWriter("predicted_video.avi", fourcc, 30, (640, 480))


# define the prediction configuration
class PredictionConfig(Config):

    # define the name of the configuration
    NAME = "part_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


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


while status:

    print(np.shape(color_image))

    try:
        # Predict for each frame -> One by one
        # Convert pixel values
        scaled_color_img = mold_image(color_image, cfg)
        # Convert image to one sample
        sample = expand_dims(scaled_color_img, 0)
        # Make the prediction
        yhat = model.detect(sample, verbose=0)
        # Get the box parameters
        y1, x1, y2, x2 = yhat[0]["rois"][0]

        # width, height = x2 - x1, y2 - y1
        # rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        rec_color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    except:

        rec_color_image = color_image

    pred_video_writer.write(rec_color_image)

    cv2.imshow("Temp", rec_color_image)

    status, color_image = video.read()

    print(count)
    count += 1

video.release()
pred_video_writer.release()




