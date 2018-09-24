#!/usr/bin/env python
# license removed for brevity
import rospy
import tf as tf2
import tensorflow as tf
import math
import numpy as np
import time
import geometry_msgs
import tf_conversions
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from numpy import linalg as LA
from tf import transformations as tfs
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


num_wps = 3
beta = 0.5
az = 0
el = 0
radius = 0
subject_pose = np.zeros((1,3))
sj_xyz = [0,0,0]
T_wc = np.zeros((1,3))
R_wc = np.zeros((1,3))

def eulerAnglesToRotationMatrix(roll, pitch, yaw):
     
    theta = [roll, pitch, yaw]

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


def Slerp_waypoints(start, end, sj):
    q1 = start - sj
    q2 = end - sj

    #print(q1, q2)    
    theta = math.acos( q1.dot(q2.transpose())[0][0]/(LA.norm(q1)*LA.norm(q2)) )
    #print(q1.dot(q2.transpose())[0][0],  (LA.norm(q1)*LA.norm(q2)), theta)
    im_waypoints = []
    for x in range(num_wps):
        scale0 = math.sin( (1-x/(1.0*num_wps))*theta )/math.sin(theta)
        scale1 = math.sin( (x/(1.0*num_wps))*theta )/math.sin(theta)
        im_wp = scale0*q2 + scale1*q1 + sj
        im_waypoints.append(im_wp)

    return im_waypoints


def calc_yaw(im_waypoint, sj_xyz):
    dist_xy = LA.norm(im_waypoint[0,0:2] - sj_xyz[0,0:2])
    cosTheta = (sj_xyz[0,0] - im_waypoint[0,0]) / dist_xy

    
    yaw = math.acos(cosTheta)
    if (sj_xyz[0,1] - im_waypoint[0,1]) < 0:
       yaw = 2*3.1415926 - yaw


    return yaw



def vp_callback(data):
    global T_wc
    global R_wc
    global sj_xyz

    vc_pose = data.data


    cx = -vc_pose[2]
    cy = -vc_pose[0]
    cz = vc_pose[1]
    flag = vc_pose[3]



    if cz < 0:
       print("cz < 0: %f", cz)
       return

 
    des_x_c = beta*cx + sj_xyz[0]
    des_y_c = beta*cy + sj_xyz[1]
    des_z_c = beta*cz + sj_xyz[2]    
   


    des_wyz_c = np.asarray([des_x_c, des_y_c, des_z_c])
    des_xyz_w = np.matmul(R_wc, des_wyz_c) + T_wc    
    sj_xyz_w = np.matmul(R_wc, np.asarray(sj_xyz).reshape(3,1) ).reshape(1,3) + T_wc    


    #print(des_xyz_w, sj_xyz_w)
    im_waypoints = Slerp_waypoints(T_wc, des_xyz_w, sj_xyz_w)
    #im_waypoints = [np.asarray([des_x, des_y, des_z]).reshape(1,3)]

    path = Path()
    for i in range(num_wps):
        im_waypoint = im_waypoints[num_wps-i-1]

        waypoint = PoseStamped()
        waypoint.pose.position.x = im_waypoint[0,0]
        waypoint.pose.position.y = im_waypoint[0,1]
        waypoint.pose.position.z = im_waypoint[0,2]
        #print(im_waypoint)

        yaw = calc_yaw(im_waypoint, sj_xyz_w)

        waypoint.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,yaw))

#        waypoint.pose.orientation.x = 0
#        waypoint.pose.orientation.y = 0
#        waypoint.pose.orientation.z = 0
#        waypoint.pose.orientation.w = 1

        path.poses.append(waypoint)

    wps = MarkerArray()

    for i in range(len(path.poses)):
        wp = path.poses[i]
	mk = Marker()
	mk.header.frame_id = "world";
	mk.header.stamp = rospy.Time.now();
	mk.type = Marker.SPHERE;
	mk.action = Marker.ADD;
	mk.pose.orientation.x = 0.0;
	mk.pose.orientation.y = 0.0;
	mk.pose.orientation.z = 0.0;
	mk.pose.orientation.w = 1.0;
	mk.scale.x = 0.15;
	mk.scale.y = 0.15;
	mk.scale.z = 0.15;
		  
	mk.ns = "wp";
	mk.color.a = 1.0;
	mk.color.r = 1;
	mk.color.g = 0;
	mk.color.b = 0;


	mk.pose.position.x = wp.pose.position.x 
	mk.pose.position.y = wp.pose.position.y 
	mk.pose.position.z = wp.pose.position.z
	mk.id = i;
	wps.markers.append(mk);

    if flag > 0:
       wp_pub.publish(path)
    visual_wp_pub.publish(wps)

def sj_callback(data):
    global sj_xyz

    sj_xyz[0] = data.data[0]
    sj_xyz[1] = data.data[1]
    sj_xyz[2] = data.data[2]
    
def odom_callback(data):
    global T_wc
    global R_wc
    odom = data.data[0:6]
    T_wc = np.asarray(odom[0:3]).reshape(1,3)
    R_wc = eulerAnglesToRotationMatrix(odom[3], -odom[4], odom[5])




if __name__ == '__main__':
    rospy.init_node('PublishWaypoints')
    wp_pub = rospy.Publisher("/waypoints",Path, queue_size = 1)
    vp_sub = rospy.Subscriber("/virtual_camera_pose", Float64MultiArray, vp_callback, queue_size = 1)
    sj_sub = rospy.Subscriber("/sj_pose", Float64MultiArray, sj_callback, queue_size = 1)
    visual_wp_pub = rospy.Publisher("/visual_wp",MarkerArray, queue_size=1)
    odom_sub = rospy.Subscriber("/sub_screen_pos", Float64MultiArray, odom_callback, queue_size = 1)


    rospy.spin()



