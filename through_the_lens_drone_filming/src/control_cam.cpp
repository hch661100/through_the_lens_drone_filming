#include <ros/ros.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64MultiArray.h"
#include <eigen3/Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/Vector3Stamped.h>
#include <std_msgs/Float64MultiArray.h>
#include "docc/basicController/PID.cpp"

#define PI 3.1415926538

// publisher handles
ros::Publisher goal_pub;
ros::Subscriber screen_sub;
ros::Subscriber odom_sub;
ros::Subscriber gimbal_sub;
ros::Subscriber target_sub;

Eigen::Quaterniond q;   
Eigen::Vector3d v;   

double orig_gyaw, orig_dyaw;
bool init_dflag = false;
bool init_gflag = false;
double prev_gimbal_yaw = 0;
double prev_gimbal_pitch = 0;
double prev_drone_yaw = 0;
double prev_drone_pitch = 0; 
double droll, dpitch, dyaw;
double groll, gpitch, gyaw;
double target_x, target_y;

PID cam_yaw_pid;


static void toEulerAngle(Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw)
{
	// roll (x-axis rotation)
	double sinr = +2.0 * (q.w() * q.x() + q.y() * q.z());
	double cosr = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
	roll = atan2(sinr, cosr);

	// pitch (y-axis rotation)
	double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
	if (fabs(sinp) >= 1)
		pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
	else
		pitch = asin(sinp);

	// yaw (z-axis rotation)
	double siny = +2.0 * (q.w() * q.z() + q.x() * q.y());
	double cosy = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());  
	yaw = atan2(siny, cosy) * 180 / PI;
}

/*

void screenCallback(const std_msgs::Float64MultiArray::ConstPtr& _msg)
{
  float dyaw = _msg->data[0];
  float dpitch = _msg->data[1];

  geometry_msgs::PoseStamped msg;
  msg.header.frame_id = "base_link";
  msg.pose.position.x = -1; 
  msg.pose.position.y = -1; 
  msg.pose.position.z = -1;   
  msg.pose.orientation.w = 10; 
  msg.pose.orientation.x = 0; 
  msg.pose.orientation.y = 0.005*dpitch; 
  msg.pose.orientation.z = 0.001*dyaw;  
  goal_pub.publish(msg);


}
*/
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
     q.x() = msg->pose.pose.orientation.x;  
     q.y() = msg->pose.pose.orientation.y;  
     q.z() = msg->pose.pose.orientation.z;  
     q.w() = msg->pose.pose.orientation.w;  

     toEulerAngle(q, droll, dpitch, dyaw);

     if(!init_dflag) {
        orig_dyaw = dyaw;
        init_dflag = true;
     }
     
     dyaw = (dyaw - orig_dyaw);

     if(dyaw - prev_drone_yaw > 100) dyaw = dyaw - 360;
     if(dyaw - prev_drone_yaw < -100) dyaw = dyaw + 360;
     
     prev_drone_yaw = dyaw;


}

void gimbalCallback(const geometry_msgs::Vector3Stamped::ConstPtr& msg){
     v.x() = msg->vector.z;
     if(!init_gflag) {
        orig_gyaw = v.x();
        init_gflag = true;
     }
     
     gyaw = (v.x() - orig_gyaw);
     gpitch = msg->vector.y;


     if(gyaw - prev_gimbal_yaw > 100) gyaw = gyaw - 360;
     if(gyaw - prev_gimbal_yaw < -100) gyaw = gyaw + 360;
     
     prev_gimbal_yaw = gyaw;

}


void targetCallback(const std_msgs::Float64MultiArray::ConstPtr& _msg){

     if(_msg->data[11] < 0.5) return;

     target_x = _msg->data[9]; target_y = _msg->data[10];

     
     //target_x = (target_x - 152)/152 * 45 + dyaw;
     //target_x = (target_x - 152)/152 * 45 + dyaw;
     //target_y = (target_y - 88)/88 * 20;
//     target_y = -(target_y - 88);
     target_y = -(target_y - 140);
     target_x = (target_x - 304);
     //std::cout<<target_x << " "<<target_y<<"  "<<_msg->data[11]<<std::endl;


  geometry_msgs::Vector3Stamped msg;
  msg.header.frame_id = "base_link";
  msg.vector.x = 0;
  msg.vector.y = 0.0015*target_y; //-0.002*( (-gpitch) - target_y); 
  //msg.vector.z = 0.04*( (gyaw) - target_x); 

  //pid 
  cam_yaw_pid.setTarget(152);
  msg.vector.z = 0.0015*target_x; //cam_yaw_pid.spin(target_x);
  //std::cout<<msg.vector.z<<std::endl;
  goal_pub.publish(msg);
}


int main(int argc, char** argv) 
{
    ros::init(argc, argv, "move_cam");
    ros::NodeHandle nh;
    ros::Rate loop_rate(100);
    goal_pub = nh.advertise<geometry_msgs::Vector3Stamped>("/djiros33/gimbal_speed_cmd", 1);
    odom_sub = nh.subscribe("/odom", 10, odomCallback);
    gimbal_sub = nh.subscribe("/djiros33/gimbal_angle", 10, gimbalCallback);
    target_sub = nh.subscribe("/sub_screen_pos", 10, targetCallback);


    cam_yaw_pid.setPID(0.004,0.0000,0.008);
cam_yaw_pid.isFlipped = false;

    while (ros::ok()) 
    {
      //publish_goal();
      ros::spinOnce();
      loop_rate.sleep();
    }
}


