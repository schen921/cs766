
<div align="center">
 <font size="20"> Object Detection for Swarm Mobile Robots using Computer Vision()
</font><br>
 <font size="14">Siyang Chen, Sina Sadeghian, Vignesh Selvaraj</font><br>
 University of Wisconsin-Madison<br>
 {schen658, ssadeghian, vselvaraj}@wisc.edu
</div>

# Introduction

Mobile robots with the capabilities to carry out fundamental manufacturing operations could havea huge impact in the way parts are produced.  Especially in the cases where conventional manufac-turing is difficult; moreover, mobile robots could collaborate with each other enabling networkedtask sharing and task scheduling .  The new ability makes this type of mobile robot superior com-pared to AGVs or mobile robots without arms, which leads to an increase in their applications inmodern industry.Obstacle and collision avoidance, object detection and estimating their size, ability to calculatecoordinates of a certain point on the object respect to its edges, ability to detected the center ofthe holes and their diameter are some of the challenges in this approach

# Interfacing with the Device

The device used for this study is a Intel Real sense Depth camera.  The camera outputs data inRGB  and  Depth.   Our  first  task  involved  interfacing  with  the  device.   We  used  ”pyrealsense2”and ”Opencv” to interface with the camera and extract the data.  The data is then packaged asa custom ROS image message as it it easy to interface with the embedded computer installed onthe robot.  The package is then sent to OpenCV for further analysis using the ”CvBridge”.  Theprocess of data acquisition and analysis is shown in Figure 1.

<div align="center">
<img src="./docs/cvbridge.png" width="400" height="200">
<br>Figure 1: The communication between ROS package and OpenCV<br>
</div>

# Object Detection Software Suite

## Implementation and Workflow
Our project employs a software suite that creates a scaleable realtime-capable object detection pipeline that is suitable for lightweight hardware with limited onboard processing power. This suite is based on Tensorflow's object detection API and supports cumstomable models that could be swapped out conveniently to fit the demand for different object detecion needs and models that are a best fit to the robot onboard hardware. The workflow of the software is shown as below:

<div align="center">
<img src="./docs/workflow.png" width="400" height="200">
<br>Figure 2: Object Detection Software Suite Workflow<br>
</div>

## Performance Optimization

- Relatively lightweight models (such as SSDLite Mobilenet) were chosen for object detection for better performance on lightweight hardware.
- These models could be swapped accordingly for better hardware available to deliver better performance (such as Faster R-CNN ResNet or EfficientDet etc).
- Capturing frames of a camera-input using OpenCV in separate threads to increase performance
- Have multithreading for the split session for better performance
- Allows models to grow memory allocation

# Circle Detection

The primary objective of this task is to detect the circles in the RGB image, which corresponds to the holes on the object. This is enabled by first converting the RGB image to grey image, followed by applying median blur to remove sharp edges and shiny objects in the image, finally, hough circle detection algorithm with hough gradient is used to identify the circles on the RGB image. The parameters for the hough circles that are detected are determined based on a trial and error method. Since the camera provides a live stream of data at 30fps, the hough algorithm is applied to each individual frame separately. The detected circles are shown in Figure 3

<div align="center">
<img src="./docs/circle_detect.gif" width="400" height="200">
<br>Figure 3: Hough Circle Detection<br>
</div>


# Calculate Hole Size

The primary objective of this task is to calculate the actual size of holes on the part using RGBand  depth  images.   First,  the  RGB  image  was  aligned  on  the  Depth  image  using  the  RealSense library.  Later,  the Hough Circle detection method was applied on the aligned RGB image,  and points on the circle were achieved.  By utilizing the depth data, converted the points on the RGB image to the cloud points.  Finally, we calculated the average distances of the corresponding points.The calculated size of the detected holes shown in Figure 4.  There is ± 0.1 mm tolerance between actual size of the hole and the predicted one.

<div align="center">
<img src="./docs/circle_size1.gif" width="400" height="200">
<br>Figure 4: Calculated size of the Hole<br>
</div>


# Results

Results



# Discussion & Future work

Discussion

# References

[1] X



