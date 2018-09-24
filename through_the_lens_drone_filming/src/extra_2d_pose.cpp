#define USE_CAFFE
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/image_encodings.h>
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "geometry_msgs/Vector3Stamped.h"


#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdio> // sscanf
#include <string> // std::string



#include <opencv2/core/core.hpp> // cv::Mat & cv::Size

#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging, CHECK, CHECK_EQ, LOG, VLOG, ...

//#include <openpose/headers.hpp>
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/core/array.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include "backward.hpp"
#include "transport_util.h"

//#include "image_recognition_msgs/PersonDetection.h"
//#include "image_recognition_msgs/BodypartDetection.h"

#include <eigen3/Eigen/Dense>
#include <Eigen/Geometry>
using namespace std;

namespace backward{
    backward::SignalHandling sh;
}
static const std::string OPENCV_WINDOW = "Openpose Window";
//static const std::string image_topic = "/videofile/image_raw";//"/usb_cam/image_raw";

// Gflags in the command line terminal. Check all the options by adding the flag `--help`, e.g. `openpose.bin --help`.
// Note: This command will show you flags for several files. Check only the flags for the file you are checking. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while 255 will not output any."
                                                        " Current OpenPose library messages are in the range 0-4: 1 for low priority messages and 4 for important ones.");
// OpenPose
DEFINE_string(model_pose,               "COCO",        "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder,             "/home/nvidia/openpose/models/",      "Folder where the pose models (COCO and MPI) are located.");
DEFINE_string(net_resolution,           "304x176",      "Multiples of 16.");
DEFINE_string(resolution,               "304x176",     "The image resolution (display). Use \"-1x-1\" to force the program to use the default images resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you want to change the initial scale, "
                                                        "you actually want to multiply the `net_resolution` by your desired initial scale.");
DEFINE_int32(num_scales,                1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it.");

DEFINE_int32(render_pose,               1,              "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering (slower but greater functionality, e.g. `alpha_X` flags). If rendering is enabled, it will render both `outputData` and `cvOutputData` with the original image and desired body part to be shown (i.e. keypoints, heat maps or PAFs).");

cv::Mat inputImage_l;
cv::Mat inputImage_r;
//image_recognition_msgs::PersonDetection person_detection_;
image_transport::Publisher img_pub;
ros::Publisher  sub_pub;

Eigen::Vector3d v;
Eigen::Quaterniond q;   
Eigen::Matrix3d R;

vector<double> sub_pos, sub_rect;

float roll, yaw, pitch;
vector<float> xyz_rpy;

cv::Mat imageDepth_;
cv::Mat debugDepth_;
int num_person = 0;
class OpenPoseNode
{
    private:
        ros::NodeHandle nh_;
        image_transport::ImageTransport* it;
        ros::Subscriber image_sub_l;
        ros::Subscriber sub_odom;
        ros::Subscriber sub_gimbal;
        ros::Subscriber image_sub_depth;

    public:
       
        bool img_l_rcv = false; 
        bool img_depth_rcv = false;
//        ros::Publisher pose_pub_l;


        cv::Mat inputImage_l;
        cv::Mat inputImage_depth;

        OpenPoseNode()
        {

          image_sub_l = nh_.subscribe("image", 1, &OpenPoseNode::convert_image_l, this);
          sub_odom = nh_.subscribe("odom", 1, &OpenPoseNode::odom_callback, this);
          sub_gimbal = nh_.subscribe("gimbal", 1, &OpenPoseNode::gimbal_callback, this);
          sub_pub = nh_.advertise<std_msgs::Float64MultiArray>("sub_screen_pos", 100);

//          pose_pub_l  = nh_.advertise<image_recognition_msgs::PersonDetection>("pose_detected_l", 50);
            
          it = new image_transport::ImageTransport(nh_);
          img_pub = it->advertise("/skeleton_img", 1);
        }

        ~OpenPoseNode(){}

        void convert_image_l(const sensor_msgs::Image msg)
        {   
            img_l_rcv = true;            
            inputImage_l = matFromImage( msg ).clone();
        }


	void gimbal_callback(const geometry_msgs::Vector3Stamped::ConstPtr& msg)
	{
	    roll = 3.1415926/180.0*msg->vector.x;
	    pitch = 3.1415926/180.0*msg->vector.y;
	    yaw = 3.1415926/180.0*msg->vector.z;
	}


        void odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
        {

	     v.x() = msg->pose.pose.position.x;
	     v.y() = msg->pose.pose.position.y;
     	     v.z() = msg->pose.pose.position.z;
	}



        cv::Mat get_l_image()
        {
            return inputImage_l;
        }
};

op::PoseModel gflagToPoseModel(const std::string& poseModeString)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (poseModeString == "COCO")
        return op::PoseModel::COCO_18;
    else if (poseModeString == "MPI")
        return op::PoseModel::MPI_15;
    else if (poseModeString == "MPI_4_layers")
        return op::PoseModel::MPI_15_4;
    else
    {
        op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
        return op::PoseModel::COCO_18;
    }
}

// Google flags into program variables
std::tuple<cv::Size, cv::Size, cv::Size, op::PoseModel> gflagsToOpParameters()
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // outputSize
    cv::Size outputSize;
    auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.width, &outputSize.height);
    op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ", __LINE__, __FUNCTION__, __FILE__);
    // netInputSize
    cv::Size netInputSize;
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.width, &netInputSize.height);
    op::checkE(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    // netOutputSize
    const auto netOutputSize = netInputSize;
    // poseModel
    const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_num_scales > 1)
        op::error("Uncompatible flag configuration: scale_gap must be greater than 0 or num_scales = 1.", __LINE__, __FUNCTION__, __FILE__);
    // Logging and return result
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    return std::make_tuple(outputSize, netInputSize, netOutputSize, poseModel);
}


void localize_subject(cv::Mat cvMatPoses, int id){
    for(int i=0; i<18; i++){
       sub_pos.push_back(cvMatPoses.at<float>(id,i,0));
       sub_pos.push_back(cvMatPoses.at<float>(id,i,1));
       sub_pos.push_back(cvMatPoses.at<float>(id,i,2));
    }
}

/*
// Wrapping the body-part-detection for publishing a ros message: Only for COCO 18 parts model now
image_recognition_msgs::PersonDetection pose_ros_wrapper(cv::Mat cvMatPoses, int id)
{
     image_recognition_msgs::PersonDetection detect_msgs;
     detect_msgs.header.stamp = ros::Time::now();
     detect_msgs.id = id;
     detect_msgs.nose.x = cvMatPoses.at<float>(id,0,0);
     detect_msgs.nose.y = cvMatPoses.at<float>(id,0,1);
     detect_msgs.nose.confidence = cvMatPoses.at<float>(id,0,2);
//     cout<<"confidence: "<<cvMatPoses.at<float>(id,0,2)<<" , y: "<<cvMatPoses.at<float>(id,0,1)<<endl;

     detect_msgs.neck.x = cvMatPoses.at<float>(id,1,0);
     detect_msgs.neck.y = cvMatPoses.at<float>(id,1,1);
     detect_msgs.neck.confidence = cvMatPoses.at<float>(id,1,2);

     detect_msgs.right_shoulder.x = cvMatPoses.at<float>(id,2,0);
     detect_msgs.right_shoulder.y = cvMatPoses.at<float>(id,2,1);
     detect_msgs.right_shoulder.confidence = cvMatPoses.at<float>(id,2,2);

     detect_msgs.right_elbow.x = cvMatPoses.at<float>(id,3,0);
     detect_msgs.right_elbow.y = cvMatPoses.at<float>(id,3,1);
     detect_msgs.right_elbow.confidence = cvMatPoses.at<float>(id,3,2);

     detect_msgs.right_wrist.x = cvMatPoses.at<float>(id,4,0);
     detect_msgs.right_wrist.y = cvMatPoses.at<float>(id,4,1);
     detect_msgs.right_wrist.confidence = cvMatPoses.at<float>(id,4,2);

     detect_msgs.left_shoulder.x = cvMatPoses.at<float>(id,5,0);
     detect_msgs.left_shoulder.y = cvMatPoses.at<float>(id,5,1);
     detect_msgs.left_shoulder.confidence = cvMatPoses.at<float>(id,5,2);

     detect_msgs.left_elbow.x = cvMatPoses.at<float>(id,6,0);
     detect_msgs.left_elbow.y = cvMatPoses.at<float>(id,6,1);
     detect_msgs.left_elbow.confidence = cvMatPoses.at<float>(id,6,2);
 
     detect_msgs.left_wrist.x = cvMatPoses.at<float>(id,7,0);
     detect_msgs.left_wrist.y = cvMatPoses.at<float>(id,7,1);
     detect_msgs.left_wrist.confidence = cvMatPoses.at<float>(id,7,2);

     detect_msgs.right_hip.x = cvMatPoses.at<float>(id,8,0);
     detect_msgs.right_hip.y = cvMatPoses.at<float>(id,8,1);
     detect_msgs.right_hip.confidence = cvMatPoses.at<float>(id,8,2);

     detect_msgs.right_knee.x = cvMatPoses.at<float>(id,9,0);
     detect_msgs.right_knee.y = cvMatPoses.at<float>(id,9,1);
     detect_msgs.right_knee.confidence = cvMatPoses.at<float>(id,9,2);

     detect_msgs.right_ankle.x = cvMatPoses.at<float>(id,10,0);
     detect_msgs.right_ankle.y = cvMatPoses.at<float>(id,10,1);
     detect_msgs.right_ankle.confidence = cvMatPoses.at<float>(id,10,2);

     detect_msgs.left_hip.x = cvMatPoses.at<float>(id,11,0);
     detect_msgs.left_hip.y = cvMatPoses.at<float>(id,11,1);
     detect_msgs.left_hip.confidence = cvMatPoses.at<float>(id,11,2);

     detect_msgs.left_knee.x = cvMatPoses.at<float>(id,12,0);
     detect_msgs.left_knee.y = cvMatPoses.at<float>(id,12,1);
     detect_msgs.left_knee.confidence = cvMatPoses.at<float>(id,12,2);

     detect_msgs.left_ankle.x = cvMatPoses.at<float>(id,13,0);
     detect_msgs.left_ankle.y = cvMatPoses.at<float>(id,13,1);
     detect_msgs.left_ankle.confidence = cvMatPoses.at<float>(id,13,2);

     detect_msgs.right_eye.x = cvMatPoses.at<float>(id,14,0);
     detect_msgs.right_eye.y = cvMatPoses.at<float>(id,14,1);
     detect_msgs.right_eye.confidence = cvMatPoses.at<float>(id,14,2);

     detect_msgs.left_eye.x = cvMatPoses.at<float>(id,15,0);
     detect_msgs.left_eye.y = cvMatPoses.at<float>(id,15,1);
     detect_msgs.left_eye.confidence = cvMatPoses.at<float>(id,15,2);

     detect_msgs.right_ear.x = cvMatPoses.at<float>(id,16,0);
     detect_msgs.right_ear.y = cvMatPoses.at<float>(id,16,1);
     detect_msgs.right_ear.confidence = cvMatPoses.at<float>(id,16,2);

     detect_msgs.left_ear.x = cvMatPoses.at<float>(id,17,0);
     detect_msgs.left_ear.y = cvMatPoses.at<float>(id,17,1);
     detect_msgs.left_ear.confidence = cvMatPoses.at<float>(id,17,2);

    return detect_msgs;
}
*/


int opRealTimeProcessing()
{

    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

    // Step 1 - Read Google flags (user defined configuration)
    cv::Size outputSize;
    cv::Size netInputSize;
    cv::Size netOutputSize;
    op::PoseModel poseModel;
    std::tie(outputSize, netInputSize, netOutputSize, poseModel) = gflagsToOpParameters();

    // Step 2 - Initialize all required classes
    op::CvMatToOpInput cvMatToOpInput{netInputSize, FLAGS_num_scales, (float)FLAGS_scale_gap};
    op::CvMatToOpOutput cvMatToOpOutput{outputSize};

    op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_num_scales, (float)FLAGS_scale_gap, poseModel,
                                              FLAGS_model_folder, FLAGS_num_gpu_start};

    op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_alpha_pose};
    op::OpOutputToCvMat opOutputToCvMat{outputSize};

    // Step 3 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)

    poseExtractorCaffe.initializationOnThread();

    poseRenderer.initializationOnThread();

    // Step 4 - Initialize the image subscriber
    OpenPoseNode opn;

    int count = 0;
    //const auto timerBegin = std::chrono::high_resolution_clock::now();
    ros::spinOnce();
    // Step 5 - Continuously process images from image subscriber
    while (ros::ok())
    {   

        const auto timerBegin = std::chrono::high_resolution_clock::now();
        // Step 6 - Get cv_image ptr and check that it is not null
        if( opn.img_l_rcv )
        {   
            cv::Mat img_l = opn.get_l_image();
            //ros::Duration(0.5).sleep();
	    xyz_rpy.clear();
	    xyz_rpy.push_back(v.x());
	    xyz_rpy.push_back(v.y());
	    xyz_rpy.push_back(v.z());
	    xyz_rpy.push_back(roll);
	    xyz_rpy.push_back(pitch);
	    xyz_rpy.push_back(yaw);
            //cv::cvtColor(img_l, img_l, CV_GRAY2BGR);
            //continue;
            // Step 7 - Format Input and Output Image
            const auto netInputArray_l = cvMatToOpInput.format(img_l);
            double scaleInputToOutput;
            op::Array<float> outputArray;
            std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(img_l);

            poseExtractorCaffe.forwardPass(netInputArray_l, img_l.size());

            const auto poseKeyPoints_l = poseExtractorCaffe.getPoseKeyPoints();

            auto person_num = poseKeyPoints_l.getSize(0);
            auto pose_num   = poseKeyPoints_l.getSize(1);
            //std::cout<<"person number: "<<person_num<<"pose number: "<<pose_num<<std::endl;
        
            if(person_num > 0){
               cv::Mat cvMatPoses = poseKeyPoints_l.getConstCvMat();

               sub_pos.clear();
               sub_rect.clear();

	       for(int i=0; i<person_num; i++){
                   //person_detection_ = pose_ros_wrapper(cvMatPoses, i);
                   localize_subject(cvMatPoses, i);
                   num_person = i+1;
	       }
//               opn.pose_pub_l.publish(person_detection_);  



               std_msgs::Float64MultiArray array;
               array.data.clear();
	       for(unsigned int i=0; i<xyz_rpy.size(); i++){
		    array.data.push_back(xyz_rpy[i]);
	       }	
               for(unsigned int i=0; i<sub_pos.size(); i++){
                    array.data.push_back(sub_pos[i]);
               }
               sub_pub.publish(array);
            }

            // Step 9 - Render the pose 
            poseRenderer.renderPose(outputArray, poseKeyPoints_l);
            // Step 10 - OpenPose output format to cv::Mat
            auto outputImage_l = opOutputToCvMat.formatToCvMat(outputArray);

            sensor_msgs::ImagePtr msg_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", outputImage_l).toImageMsg();
            img_pub.publish(msg_img);
            
            opn.img_l_rcv = false;
            opn.img_depth_rcv = false;
        }        
        ros::spinOnce();
        const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-timerBegin).count() * 1e-9;
        cout<<"Runtime: "<<totalTimeSec<<endl;
    }

    // Measuring total time
//    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-timerBegin).count() * 1e-9;
//    cout<<"Runtime: "<<totalTimeSec<<endl;
//    const auto message = "Real-time pose estimation demo successfully finished. Total time: " + std::to_string(totalTimeSec) + " seconds. " 
//                         + std::to_string(count) + " frames processed. Average FPS is " + std::to_string(count/totalTimeSec);
//    op::log(message, op::Priority::Max);

    return 0;
}

ros::Subscriber image_sub_;
int main(int argc, char** argv)
{
    //google::InitGoogleLogging("openpose_ros_node");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ros::init(argc, argv, "extra_2d_pose");
  
    return opRealTimeProcessing();
}
