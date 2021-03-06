# Through-the-Lens Drone Filming

"Through-the-Lens Drone Filming" represents a novel drone control system, which allows a cameraman to conveniently control the drone by manipulating a 3D model in the preview, which closes the gap between the ﬂight control and the viewpoint design. 

Our system includes two key enabling techniques: 1) subject localization based on visual-inertial fusion, and 2) through-the-lens camera planning. This is the ﬁrst drone camera system which allows users to capture human actions by manipulating the camera in a virtual environment. 

<p align="center">
<img src="https://github.com/hch661100/through_the_lens_drone_filming/blob/master/through_the_lens_drone_filming/resource/ui2.png" width="400">
</p>

The code is required to be deployed on Android device and DJI M100 (including Zenmuse X3). 

### Preparation	     
We mount Jetson TX2 and DJI Manifold on the DJI M100. The DJI Manifold is used to decode the video from Zenmuse X3 and run the flight control. The subject localization algorithm is depolyed on Jetson TX2. Make sure that you already installed the packages as follows:

Manifold:   https://github.com/hch661100/Manifold.
TX2:   https://github.com/hch661100/tx2.

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
Estimate the absolute scale of the subject:
```
rosrun through_the_lens_drone_filming scale_estimation
```
Launch drone control:
```
roslaunch through_the_lens_drone_filming  through_the_lens_drone_filming.launch
```
### Citation
Please cite this paper in your publications if it helps your research.
```
@article{huangthrough,
  title={Through-the-Lens Drone Filming},
  author={Huang, Chong and Yang, Zhenyu and Kong, Yan and Chen, Peng and Yang, Xin and Cheng, Kwang-Ting Tim}
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year = {2018}
}
```
