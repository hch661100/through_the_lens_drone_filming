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
from numpy.linalg import inv
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

num_wps = 3
TIMESTEP = 5
BATCH_SIZE = 32 #how many samples to use in one training epoch
INPUT_SIZE = 28
OUTPUT_SIZE = 42
CELL_SIZE = 42
DROPOUT_PROB =  1.0
LR = 1e-5
count = 0

limb_dict = [33, 36, 39, 15, 18, 21, 3, 0, 6, 9, 12, 24, 27, 30,
             34, 37, 40, 16, 19, 22, 4, 1, 7,10, 13, 25, 28, 31]

joint_id = [0, 1, 1, 2, 3, 4, 4, 5, 6, 3, 6, 7, 6, 8, 8, 9, 9, 10, 6, 11, 11, 12, 12, 13, 6, 0]

#v_t = [1,-4,1.5,    0,3,0.5,   6,6,0.5,  -4,7,0.5,  0,3,1.5]
#v_e = [5,0,1.5,     8,0,0.5,   1,4,0.5,  0,4,0.5,   10,0,1]
#v_d = [1,4.89,4,    4,-4,0.5,  -6,6,1.5,    8,7,0.5,  4,-4,0.5]

v_t = [0,3,1.5,   1,0,0.5,    2.5,0,0.5]
v_e = [10,0,1,    3,0,1.5,    4,4,0.3]
v_d = [4,-4,0.5,  5,0,5,      8,8,3]

ind_v = 0

def weight_variable(shape): #define neural network weight
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):# define neural network bias
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class RNNPOSE(object):
  def __init__(self, input_size, output_size, linear_size, dropout_keep_prob, batch_size):
    self.input_size = input_size
    self.output_size = output_size
    self.linear_size = linear_size
    self.dropout_keep_prob = dropout_keep_prob
    self.batch_size = batch_size

    with tf.name_scope('inputs'):
       self.xy = tf.placeholder(tf.float32, shape=[None, None, 28]) #input placeholder in tensorflow
       #self.xyz = tf.placeholder(tf.float32, shape=[None, None, 42]) #ground truth placeholder
    with tf.name_scope('encoder'):
       self.cell1 = tf.contrib.rnn.LayerNormBasicLSTMCell(linear_size,dropout_keep_prob=self.dropout_keep_prob)
       self.cell1 = tf.contrib.rnn.DropoutWrapper(self.cell1,input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)
       self.cell1 = tf.contrib.rnn.InputProjectionWrapper(self.cell1,linear_size)
       self.cell1 = tf.contrib.rnn.OutputProjectionWrapper(self.cell1,self.input_size)
    with tf.name_scope('decoder'):
       self.cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(linear_size,dropout_keep_prob=self.dropout_keep_prob)
       self.cell2 = tf.contrib.rnn.DropoutWrapper(self.cell2,input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)
       self.cell2 = tf.contrib.rnn.InputProjectionWrapper(self.cell2,linear_size)
       self.cell2 = tf.contrib.rnn.OutputProjectionWrapper(self.cell2,self.output_size)
       self.cell2 = tf.contrib.rnn.ResidualWrapper(self.cell2)
    with tf.name_scope('test'):
       self.validate()


  def validate(self):

       def lf(prev, i):
           return prev

       dec_outputs = []
       un_norm_dec_gt = []


       for x in range(1):

           enc_outputs = []
           enc_state = []
           enc_in = []
           state = self.cell1.zero_state(1,dtype=tf.float32)


           for t in range(TIMESTEP):
             enc_in.append(self.xy[:,x+t,:])
             #un_norm_dec_gt.append(self.xyz[:,x+t,:])

           for inputs in enc_in:
             out,state = self.cell1(inputs,state)
             enc_outputs.append(out)
             enc_state.append(state)

           go_symbol = []
           for t in range(TIMESTEP):
             if t == 0:
                go_symbol.append(tf.ones([1,42],dtype=tf.float32))
                dec_in = go_symbol
                dec_state = enc_state[-1]
             else:
                self.debug = dec_in
                dec_in = dec_in_

             dec_out, dec_state = tf.contrib.legacy_seq2seq.rnn_decoder( dec_in, dec_state, self.cell2, loop_function=lf )

             dec_in_ = []
             dec_xy = tf.reshape(enc_in[t],[1,28])
             dec_z = dec_out[0][:, 28:42]     
             dec_xyz = tf.concat([dec_xy, dec_z],1)   
             dec_in_.append(dec_xyz)
             dec_outputs.append(dec_xyz)

       self.pred3d = dec_outputs[-1] 
       
       #un_norm_out = dec_outputs
       #un_norm_out = tf.stack(dec_outputs)

       #self.test_err = tf.reduce_mean(tf.square(tf.subtract(un_norm_dec_gt ,un_norm_out)))


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



def norm_height(test_x, test_y, test_z):
    pose3d = np.concatenate((test_x, test_y, test_z)).reshape(3,14)
    height = LA.norm(pose3d[:,7] - pose3d[:,6]) + 0.5*(LA.norm(pose3d[:,6] - pose3d[:,11]) + LA.norm(pose3d[:,6] - pose3d[:,0]) + LA.norm(pose3d[:,0] - pose3d[:,1]) + LA.norm(pose3d[:,1] - pose3d[:,2])+LA.norm(pose3d[:,11] - pose3d[:,12]) + LA.norm(pose3d[:,12] - pose3d[:,13]))
    coff = 1.8/height
    test_x = coff*test_x
    test_y = coff*test_y
    test_z = coff*test_z
    return test_x, test_y, test_z

def localize_subject(norm_x, norm_y, norm_z, pixel_x, pixel_y):

    ratio = 640/608
    pixel_x *= ratio
    pixel_y *= ratio

    pixel_x -= 320
    pixel_y -= 180

    
    norm_xy = np.concatenate((norm_x, norm_y)).reshape(2,14)
    pixel_xy = np.concatenate((pixel_x, pixel_y)).reshape(2,14)
    fx = 280.917376
    fy = 280.938199
    F = np.array([[fx, 0], [0, fy]])
    mean_uv = np.array([np.mean(pixel_x),np.mean(pixel_y)])
    mean_xy = np.array([np.mean(norm_x),np.mean(norm_y)])   
    numerator = 0
    denominator = 0
    for i in xrange(0, 14):  
       numerator += np.power(LA.norm(F*(norm_xy[:,i] - mean_xy)),2)
       denominator += LA.norm( np.transpose((pixel_xy[:,i] - mean_uv))*F*(norm_xy[:,i] - mean_xy) )

    tz = numerator / denominator
    tx =  np.mean(pixel_x)*tz/fx - np.mean(norm_x)
    ty =  np.mean(pixel_y)*tz/fy - np.mean(norm_y)

    norm_x += tx
    norm_y += ty
    norm_z += tz

    return norm_x, norm_y, norm_z, tx, ty, tz

def calc_yaw(im_waypoint, sj_xyz):
    dist_xy = LA.norm(im_waypoint[0,0:2] - sj_xyz[0,0:2])
    cosTheta = (sj_xyz[0,0] - im_waypoint[0,0]) / dist_xy

    
    yaw = math.acos(cosTheta)
    if (sj_xyz[0,1] - im_waypoint[0,1]) < 0:
       yaw = 2*3.1415926 - yaw


    return yaw



class PoseEstimation():
    def __init__(self):

        self.model = RNNPOSE(INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, DROPOUT_PROB, BATCH_SIZE)
        self.br = tf2.TransformBroadcaster()
        self.xy_ = self.model.xy #input placeholder in tensorflow
        self.z_ = self.model.pred3d 
        self._saver = tf.train.Saver()
        self._session = tf.InteractiveSession()
        self.prev_pose2d = np.zeros((2,14))
        init_op = tf.global_variables_initializer()
        self._session.run(init_op)

        self._saver.restore(self._session, "/home/hch/Project/tf/model/model_rnn/model_weight_wot_200.ckpt")
        self.stack = []
        self._sub = rospy.Subscriber("/sub_screen_pos", Float64MultiArray, self.callback, queue_size = 1)
        self._pub3d = rospy.Publisher("/pose3d",MarkerArray, queue_size=1)
        self._pub3d_sa = rospy.Publisher("/subject_axis",MarkerArray, queue_size=1)
        self._pub_sj_position = rospy.Publisher("/sj_pose",Float64MultiArray, queue_size=1)
        self._pub_norm_3d = rospy.Publisher("/djiros33/norm3d",Float32MultiArray, queue_size=1)

    def callback(self, data):
        global count
        pose2d_odom = data.data

########### Send the odom ########################
        odom = pose2d_odom[0:6]

        wRc = eulerAnglesToRotationMatrix(odom[3],-odom[4],odom[5])
        wTc = np.asarray(odom[0:3]).reshape(1,3)


	pose2d = pose2d_odom[6:60]
        test_xyc = (np.float32(pose2d)).reshape(1,54)
########### Send the 2D Pose #####################

        pixel_xy = test_xyc[:,limb_dict]


        missing_xy = np.where(pixel_xy == 0)[1]

        for x in range(len(missing_xy)):                    
            pixel_xy[0, missing_xy[x]] = self.prev_pose2d[0, missing_xy[x]]

        self.prev_pose2d = np.copy(pixel_xy)



        pixel_x = pixel_xy[:,0:14]
        pixel_y = pixel_xy[:,14:28]
        mean_x = np.mean(pixel_x)
        mean_y = np.mean(pixel_y)
        std_x = np.std(pixel_x)
        std_y = np.std(pixel_y)
        test_x = (pixel_x - mean_x)/((std_x + std_y)/2)
        test_y = (pixel_y - mean_y)/((std_x + std_y)/2)
        test_xy = np.concatenate((test_x, test_y)).reshape(1,28)

        self.stack.append(test_xy)
        start_time = time.time()


        if(len(self.stack) < 5):
          return
        
        if(len(self.stack) == 5):
          data = np.asarray(self.stack).reshape(1,5,28)
          output = self._session.run(self.z_, feed_dict={self.xy_:data})
          self.stack.pop(0)

        msg_float_array = Float32MultiArray()
        msg_float_array.data = output.reshape(1,42).tolist()[0]
        self._pub_norm_3d.publish(msg_float_array)


        elapsed_time = time.time() - start_time


        output = output.reshape(3,14)
        test_x = output[0,:] 
        test_y = output[1,:] 
        test_z = -output[2,:] ########## test_z = -output[2,:] when the Openpose is used


        norm_x, norm_y, norm_z = norm_height(test_x, test_y, test_z)
        norm_x, norm_y, norm_z, tx, ty, tz = localize_subject(norm_x, norm_y, norm_z, pixel_x, pixel_y)


        norm_pose_ = np.asarray([norm_z, -norm_x, -norm_y]).reshape(3,14)
        norm_pose = np.matmul(wRc, norm_pose_) + np.matlib.repmat(wTc.reshape(3,1), 1, 14)


        norm_x = norm_pose[0,:]
        norm_y = norm_pose[1,:]
        norm_z = norm_pose[2,:]

  
        y_axis_s = ((norm_pose[:,3] - norm_pose[:,8])/LA.norm(norm_pose[:,3] - norm_pose[:,8])).reshape(1,3)
        z_axis_s = np.asarray([0,0,1]).reshape(1,3)
        x_axis_s = np.cross(y_axis_s, z_axis_s)
        y_axis_s = np.cross(z_axis_s, x_axis_s)

        y_axis_s_ = ((norm_pose_[:,3] - norm_pose_[:,8])/LA.norm(norm_pose_[:,3] - norm_pose_[:,8])).reshape(1,3)
        z_axis_s_ = np.asarray([0,0,1]).reshape(1,3)
        x_axis_s_ = np.cross(y_axis_s_, z_axis_s_)
        y_axis_s_ = np.cross(z_axis_s_, x_axis_s_)

        R_sc = np.asarray([x_axis_s_, y_axis_s_, z_axis_s_]).reshape(3,3).transpose()

        msg_float_array = Float64MultiArray()
        subject_position_c = [tz, -tx, -ty]
        msg_float_array.data = subject_position_c
        self._pub_sj_position.publish(msg_float_array)

        subject_position_w = np.matmul(wRc, np.asarray(subject_position_c).reshape(3,1)) + np.matlib.repmat(wTc.reshape(3,1), 1, 14)

        viewpoint_sa = np.asarray(v_t[3*ind_v:3*ind_v+3]).reshape(3,1)
        viewpoint_c = np.matmul(R_sc, viewpoint_sa) + np.asarray(subject_position_c).reshape(3,1)
        viewpoint_position_w0 = np.matmul(wRc, np.asarray(viewpoint_c).reshape(3,1) ) + np.matlib.repmat(wTc.reshape(3,1), 1, 14)
 
        viewpoint_sa = np.asarray(v_e[3*ind_v:3*ind_v+3]).reshape(3,1)
        viewpoint_c = np.matmul(R_sc, viewpoint_sa) + np.asarray(subject_position_c).reshape(3,1)
        viewpoint_position_w1 = np.matmul(wRc, np.asarray(viewpoint_c).reshape(3,1) ) + np.matlib.repmat(wTc.reshape(3,1), 1, 14)

        viewpoint_sa = np.asarray(v_d[3*ind_v:3*ind_v+3]).reshape(3,1)
        viewpoint_c = np.matmul(R_sc, viewpoint_sa) + np.asarray(subject_position_c).reshape(3,1)
        viewpoint_position_w2 = np.matmul(wRc, np.asarray(viewpoint_c).reshape(3,1) ) + np.matlib.repmat(wTc.reshape(3,1), 1, 14)

        path = Path()

        ##### v_d ######
        waypoint = PoseStamped()
        waypoint.pose.position.x = viewpoint_position_w0[0,0]
        waypoint.pose.position.y = viewpoint_position_w0[1,0]
        waypoint.pose.position.z = viewpoint_position_w0[2,0]

        yaw = calc_yaw(np.asarray(viewpoint_position_w0[:,0]).reshape(1,3), subject_position_w[:,0].reshape(1,3))
        waypoint.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,yaw))
        path.poses.append(waypoint)


        ##### v_e ######
        waypoint = PoseStamped()
        waypoint.pose.position.x = viewpoint_position_w1[0,0]
        waypoint.pose.position.y = viewpoint_position_w1[1,0]
        waypoint.pose.position.z = viewpoint_position_w1[2,0]

        yaw = calc_yaw(np.asarray(viewpoint_position_w1[:,0]).reshape(1,3), subject_position_w[:,0].reshape(1,3))
        waypoint.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,yaw))
        path.poses.append(waypoint)

        ##### v_d ######
        waypoint = PoseStamped()
        waypoint.pose.position.x = viewpoint_position_w2[0,0]
        waypoint.pose.position.y = viewpoint_position_w2[1,0]
        waypoint.pose.position.z = viewpoint_position_w2[2,0]

        yaw = calc_yaw(np.asarray(viewpoint_position_w2[:,0]).reshape(1,3), subject_position_w[:,0].reshape(1,3))
        waypoint.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,yaw))
        path.poses.append(waypoint)
        


        count = count + 1

        if count == 10:
           self._wp_pub.publish(path)
           #count = 0

####################################################

        pose3d = MarkerArray()
        subject_axis = MarkerArray()


        cam_track = Marker()
        cam_track.header.frame_id = "world";
        cam_track.header.stamp = rospy.Time.now();
        cam_track.type = Marker.SPHERE;
        cam_track.action = Marker.ADD;
        cam_track.pose.orientation.x = 0.0;
        cam_track.pose.orientation.y = 0.0;
        cam_track.pose.orientation.z = 0.0;
        cam_track.pose.orientation.w = 1.0;
        cam_track.scale.x = 0.15;
        cam_track.scale.y = 0.15;
        cam_track.scale.z = 0.15;
		  
        cam_track.ns = "cam_track";
        cam_track.color.a = 1.0;
        cam_track.color.r = 0;
        cam_track.color.g = 0.5;
        cam_track.color.b = 0.5;


        cam_track.pose.position.x = viewpoint_position_w1[0,0]
        cam_track.pose.position.y = viewpoint_position_w1[1,0]
        cam_track.pose.position.z = viewpoint_position_w1[2,0]
        cam_track.id = 0;
        subject_axis.markers.append(cam_track);


        cam_track = Marker()
        cam_track.header.frame_id = "world";
        cam_track.header.stamp = rospy.Time.now();
        cam_track.type = Marker.SPHERE;
        cam_track.action = Marker.ADD;
        cam_track.pose.orientation.x = 0.0;
        cam_track.pose.orientation.y = 0.0;
        cam_track.pose.orientation.z = 0.0;
        cam_track.pose.orientation.w = 1.0;
        cam_track.scale.x = 0.15;
        cam_track.scale.y = 0.15;
        cam_track.scale.z = 0.15;
	  
        cam_track.ns = "cam_track";
        cam_track.color.a = 1.0;
        cam_track.color.r = 0;
        cam_track.color.g = 1.0;
        cam_track.color.b = 0.0;

        cam_track.pose.position.x = viewpoint_position_w2[0,0]
        cam_track.pose.position.y = viewpoint_position_w2[1,0]
        cam_track.pose.position.z = viewpoint_position_w2[2,0]
        cam_track.id = -1;
        subject_axis.markers.append(cam_track);


        ps = Point()
        ps.x = subject_position_w[0,0]
        ps.y = subject_position_w[1,0]
        ps.z = subject_position_w[2,0]

        axis_x = Marker()
        axis_x.points = []
        axis_x.header.frame_id = "world";
        axis_x.header.stamp = rospy.Time.now();
        axis_x.ns = "subject_coordinates";

        axis_x.action = Marker.ADD;
        axis_x.pose.orientation.w = 1.0;
        axis_x.type = Marker.LINE_LIST;

        axis_x.id = 0;
        axis_x.scale.x = 0.1;
        axis_x.scale.y = 0.1;
        axis_x.scale.z = 0.1;
	  
        axis_x.color.a = 1.0;
        axis_x.color.r = 1.0;
        axis_x.color.g = 0.0 ;
        axis_x.color.b = 0.0;

        px = Point()
        px.x = x_axis_s[0,0]+subject_position_w[0,0]
        px.y = x_axis_s[0,1]+subject_position_w[1,0]
        px.z = x_axis_s[0,2]+subject_position_w[2,0]
        axis_x.points.append(ps);
        axis_x.points.append(px);
        subject_axis.markers.append(axis_x)

        axis_y = Marker()
        axis_y.points = []
        axis_y.header.frame_id = "world";
        axis_y.header.stamp = rospy.Time.now();
        axis_y.ns = "subject_coordinates";

        axis_y.action = Marker.ADD;
        axis_y.pose.orientation.w = 1.0;
        axis_y.type = Marker.LINE_LIST;

        axis_y.id = 1;
        axis_y.scale.x = 0.1;
        axis_y.scale.y = 0.1;
        axis_y.scale.z = 0.1;
	  
        axis_y.color.a = 1.0;
        axis_y.color.r = 0.0;
        axis_y.color.g = 1.0 ;
        axis_y.color.b = 0.0;

        py = Point()
        py.x = y_axis_s[0,0]+subject_position_w[0,0]
        py.y = y_axis_s[0,1]+subject_position_w[1,0]
        py.z = y_axis_s[0,2]+subject_position_w[2,0]
        axis_y.points.append(ps);
        axis_y.points.append(py);
        subject_axis.markers.append(axis_y)

        axis_z = Marker()
        axis_z.points = []
        axis_z.header.frame_id = "world";
        axis_z.header.stamp = rospy.Time.now();
        axis_z.ns = "subject_coordinates";

        axis_z.action = Marker.ADD;
        axis_z.pose.orientation.w = 1.0;
        axis_z.type = Marker.LINE_LIST;

        axis_z.id = 2;
        axis_z.scale.x = 0.1;
        axis_z.scale.y = 0.1;
        axis_z.scale.z = 0.1;
	  
        axis_z.color.a = 1.0;
        axis_z.color.r = 0.0;
        axis_z.color.g = 0.0 ;
        axis_z.color.b = 1.0;

        pz = Point()
        pz.x = z_axis_s[0,0]+subject_position_w[0,0]
        pz.y = z_axis_s[0,1]+subject_position_w[1,0]
        pz.z = z_axis_s[0,2]+subject_position_w[2,0]
        axis_z.points.append(ps);
        axis_z.points.append(pz);
        subject_axis.markers.append(axis_z)
        self._pub3d_sa.publish(subject_axis)
        

        line_inf = Marker()
        line_inf.points = []
        line_inf.header.frame_id = "world";
        line_inf.header.stamp = rospy.Time.now();
        line_inf.ns = "pose";

        line_inf.action = Marker.ADD;
        line_inf.pose.orientation.w = 1.0;
        line_inf.type = Marker.LINE_LIST;

        line_inf.id = 100;
        line_inf.scale.x = 0.1;
        line_inf.scale.y = 0.1;
        line_inf.scale.z = 0.1;
	  
        line_inf.color.a = 1.0;
        line_inf.color.r = 0.5;
        line_inf.color.g = 0.5 ;
        line_inf.color.b = 0.0;

        norm_x = norm_x.reshape(1,14)
        norm_y = norm_y.reshape(1,14)
        norm_z = norm_z.reshape(1,14)

        for i in xrange(0, 13):
           p1 = Point()
           ind1 = joint_id[2*i];
           p1.x = norm_x.item((0, ind1)) 
           p1.y = norm_y.item((0, ind1)) 
           p1.z = norm_z.item((0, ind1))
           line_inf.points.append(p1);

           p2 = Point()
           ind2 = joint_id[2*i+1];
           p2.x = norm_x.item((0, ind2)) 
           p2.y = norm_y.item((0, ind2)) 
           p2.z = norm_z.item((0, ind2))
           line_inf.points.append(p2);

        pose3d.markers.append(line_inf);


        for i in xrange(0, 14):

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
		  
          mk.ns = "pose";
          mk.color.a = 1.0;
          mk.color.r = 0.5;
          mk.color.g = 0.5;
          mk.color.b = 0;

          mk.pose.position.x = norm_x.item((0, i)) 
          mk.pose.position.y = norm_y.item((0, i)) 
          mk.pose.position.z = norm_z.item((0, i))
          mk.id = i;
          pose3d.markers.append(mk);


        self._pub3d.publish(pose3d)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('2d_map_3d')
    pe = PoseEstimation()
    pe.main()

