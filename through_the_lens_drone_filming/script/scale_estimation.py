#!/usr/bin/env python
# license removed for brevity
import rospy
import tf as tf2
import tensorflow as tf
import math
import numpy as np
import time
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from numpy import linalg as LA
from tf import transformations as tfs
from numpy.linalg import inv


TIMESTEP = 5
BATCH_SIZE = 32 #how many samples to use in one training epoch
INPUT_SIZE = 28
OUTPUT_SIZE = 42
CELL_SIZE = 42
DROPOUT_PROB =  1.0
LR = 1e-5
subject_height = 1.8

limb_dict = [33, 36, 39, 15, 18, 21, 3, 0, 6, 9, 12, 24, 27, 30,
             34, 37, 40, 16, 19, 22, 4, 1, 7,10, 13, 25, 28, 31]

joint_id = [0, 1, 1, 2, 3, 4, 4, 5, 6, 3, 6, 7, 6, 8, 8, 9, 9, 10, 6, 11, 11, 12, 12, 13, 6, 0]

height_seq = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]

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
             #   dec_in = []
             #   dec_xy = tf.reshape(enc_in[t-1],[1,28])
             #   dec_z = dec_out[0][:, 28:42]        
             #   dec_xyz = tf.concat([dec_xy, dec_z],1)
             #   dec_in.append(dec_xyz)
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
    coff = 1.0/height
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

        self._saver.restore(self._session, "/home/hch/Project/tf/model/model_rnn/model_weight_wot2_200.ckpt")
        self.stack = []
        self.obs = []
        self._sub = rospy.Subscriber("/sub_screen_pos", Float64MultiArray, self.callback, queue_size = 1)
        self._pub_norm_3d = rospy.Publisher("/djiros33/norm3d",Float32MultiArray, queue_size=1)


    def callback(self, data):
        pose2d_odom = data.data

########### Send the odom ########################
	odom = pose2d_odom[0:6]

        wRc = eulerAnglesToRotationMatrix(odom[3],-odom[4],odom[5])
        wTc =  np.asarray(odom[0:3]).reshape(1,3)

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

        if(len(self.stack) < 5):
          return
        
        if(len(self.stack) == 5):
          data = np.asarray(self.stack).reshape(1,5,28)
          output = self._session.run(self.z_, feed_dict={self.xy_:data})
          self.stack.pop(0)

        msg_float_array = Float32MultiArray()
        msg_float_array.data = output.reshape(1,42).tolist()[0]
        self._pub_norm_3d.publish(msg_float_array)


        output = output.reshape(3,14)
        test_x = output[0,:] 
        test_y = output[1,:] 
        test_z = -output[2,:]

        norm_x, norm_y, norm_z = norm_height(test_x, test_y, test_z)

        if len(self.obs) < 11:
           self.obs.append([test_x, test_y, test_z, pixel_x, pixel_y])
        print len(self.obs)

        if len(self.obs) == 10:
          opt_height = []
          for height in height_seq:
            std_pos = []
            for nx,ny,nz,px,py in self.obs:
              norm_x = height * nx
              norm_y = height * ny
              norm_z = height * nz
              norm_x, norm_y, norm_z, tx, ty, tz = localize_subject(norm_x, norm_y, norm_z, px, py)
              std_pos.append([tx, ty, tz])              
            opt_height.append(LA.norm(np.std(std_pos, axis=0)))
          opt_height = height_seq[np.argmin(opt_height)]
          print opt_height



    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('PoseEstimation')
    pe = PoseEstimation()
    pe.main()

