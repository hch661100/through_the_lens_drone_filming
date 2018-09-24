# Through-the-Lens Drone Filming

[![Watch the video](https://raw.github.com/GabLeRoux/WebMole/master/ressources/WebMole_Youtube_Video.png)](http://youtu.be/vt5fpE0bzSY)


"Through-the-Lens Drone Filming" represents a novel drone control system, which allows a cameraman to conveniently control the drone by manipulating a 3D model in the preview, which closes the gap between the ﬂight control and the viewpoint design. 

Our system includes two key enabling techniques: 1) subject localization based on visual-inertial fusion, and 2) through-the-lens camera planning. This is the ﬁrst drone camera system which allows users to capture human actions by manipulating the camera in a virtual environment. 

The code is required to be deployed on Android device and DJI M100 (including Zenmuse X3). 

### Preparation	     
We mount Jetson TX2 and DJI Manifold on the DJI M100. The DJI Manifold is used to decode the video from Zenmuse X3 and run the flight control. Make sure that you already installed the package in https://github.com/hch661100/Manifold.

The subject localization algorithm is depolyed on Jetson TX2. Before running the algorithm, and make sure that you set up the TX2 in https://github.com/hch661100/tx2.

### Installation     
Catkin Build on TX2:
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/hch661100/through_the_lens_drone_filming.git
$ cd ..
$ catkin_make
```
Android code: https://github.com/hch661100/through_the_lens_drone_filming/tree/master/Android

### Quick Start
2D skeleton:
```
roslaunch through_the_lens_drone_filming extra_2d_pose.launch
```
launch drone control:
```
roslaunch through_the_lens_drone_filming  through_the_lens_drone_filming.launch
```
