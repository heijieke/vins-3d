#include <ros/ros.h>
#include <glog/logging.h>
#include <atomic>
#include "libobsensor/ObSensor.hpp"
#include "orbbec_camera/Extrinsics.h"
#include "orbbec_camera/GetCameraInfo.h"

#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include "utils.h"
#include <thread>
#include <image_transport/publisher.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <condition_variable>


#define STREAM_NAME(sip)                                                                       \
  (static_cast<std::ostringstream&&>(std::ostringstream()                                      \
                                     << _stream_name[sip.first]                                \
                                     << ((sip.second > 0) ? std::to_string(sip.second) : ""))) \
      .str()
#define FRAME_ID(sip)                                                                              \
  (static_cast<std::ostringstream&&>(std::ostringstream()                                          \
                                     << getNamespaceStr() << "_" << STREAM_NAME(sip) << "_frame")) \
      .str()
#define OPTICAL_FRAME_ID(sip)                                                                     \
  (static_cast<std::ostringstream&&>(                                                             \
       std::ostringstream() << getNamespaceStr() << "_" << STREAM_NAME(sip) << "_optical_frame")) \
      .str()
#define ALIGNED_DEPTH_TO_FRAME_ID(sip)                                            \
  (static_cast<std::ostringstream&&>(std::ostringstream()                         \
                                     << getNamespaceStr() << "_aligned_depth_to_" \
                                     << STREAM_NAME(sip) << "_frame"))            \
      .str()
#define BASE_FRAME_ID() \
  (static_cast<std::ostringstream&&>(std::ostringstream() << getNamespaceStr() << "_link")).str()
#define ODOM_FRAME_ID()                                                                           \
  (static_cast<std::ostringstream&&>(std::ostringstream() << getNamespaceStr() << "_odom_frame")) \
      .str()

typedef std::pair<ob_stream_type, int> stream_index_pair;

const stream_index_pair COLOR{OB_STREAM_COLOR, 0};
const stream_index_pair DEPTH{OB_STREAM_DEPTH, 0};
const stream_index_pair INFRA0{OB_STREAM_IR, 0};
const stream_index_pair INFRA1{OB_STREAM_IR, 1};
const stream_index_pair INFRA2{OB_STREAM_IR, 2};

const stream_index_pair GYRO{OB_STREAM_GYRO, 0};
const stream_index_pair ACCEL{OB_STREAM_ACCEL, 0};

const std::vector<stream_index_pair> IMAGE_STREAMS = {DEPTH, INFRA0, COLOR};
const std::vector<stream_index_pair> HID_STREAMS = {GYRO, ACCEL};

class OBCamera{
    public:
        OBCamera(ros::NodeHandle& nh,const std::shared_ptr<ob::Device> device);
        ~OBCamera();
        void clean();
    private:
        ros::NodeHandle& nh_;
        const std::shared_ptr<ob::Device> device_;
        std::unique_ptr<ob::Pipeline> pipeline_ = nullptr;
        std::shared_ptr<ob::Sensor> accelSensor_ = nullptr;
        std::shared_ptr<ob::Sensor> gyroSensor_ = nullptr;
        std::shared_ptr<ob::Config> config_ = nullptr;
        std::map<stream_index_pair, std::vector<std::shared_ptr<ob::VideoStreamProfile>>> enabled_profiles_;
        std::map<stream_index_pair, std::shared_ptr<ob::Sensor>> sensors_;
        std::vector<std::pair<std::string, int>> param_;
        std::string d2c_mode_;
        std::string camera_link_frame_id_;
        int color_width_;
        int color_height_;
        int color_fps_;
        int ir_width_;
        int ir_height_;
        int ir_fps_;
        int depth_width_;
        int depth_height_;
        int depth_fps_;
        uint32_t last_time = 0;
        uint32_t max_time = 99999999/10;
        bool align_depth_;
        bool publish_tf_;
        double tf_publish_rate_ = 10.0;
        bool publish_rgb_point_cloud_;
        std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
        std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_tf_broadcaster_;
        std::vector<geometry_msgs::TransformStamped> static_tf_msgs_;
        std::shared_ptr<std::thread> tf_thread_;
        std::atomic_bool is_running_{false};
        std::map<stream_index_pair, bool> enable_;
        std::vector<int> compression_params_;
        std::map<stream_index_pair, int> unit_step_size_;
        std::map<stream_index_pair, int> width_;
        std::map<stream_index_pair, int> height_;
        std::map<stream_index_pair, double> fps_;
        std::map<stream_index_pair, std::string> frame_id_;
        std::map<stream_index_pair, std::string> optical_frame_id_;
        std::map<stream_index_pair, std::string> depth_aligned_frame_id_;
        std::map<ob_stream_type, std::string> stream_name_;
        std::map<stream_index_pair, std::string> encoding_;
        std::map<stream_index_pair, ob_format> format_;
        std::map<stream_index_pair, std::string> format_str_;
        std::map<ob_stream_type, int> image_format_;
        std::map<stream_index_pair, cv::Mat> images_;
        sensor_msgs::Imu imu_msg_;
        sensor_msgs::PointCloud2 point_cloud_msg_;
        std::map<stream_index_pair, sensor_msgs::CameraInfo> camera_infos_;
        std::map<stream_index_pair, image_transport::Publisher> image_publishers_;
        image_transport::Publisher imageG_publishers_;
        std::map<stream_index_pair, ros::Publisher> camera_info_publishers_;

        ros::Publisher imu_publisher_;
        ros::Publisher point_cloud_publisher_;
        ros::Publisher depth_point_cloud_publisher_;
        ros::Publisher extrinsics_publisher_;
        sensor_msgs::CameraInfo depth_cameraInfo_;
        sensor_msgs::CameraInfo color_cameraInfo_;

        ob::FormatConvertFilter format_convert_filter_;
        ob::PointCloudFilter point_cloud_filter_;
        std::condition_variable tf_cv_;

        void setupDevices();
        void setupProfiles();
        void startPipeline();
        void setupPublishers();
        void frameSetCallback(std::shared_ptr<ob::FrameSet> frame_set);
        void updateStreamCalibData(const OBCameraParam& param);
        void publishColorFrame();
        void publishDepthFrame();
        void publishIRFrame();
        void publishPointCloud();
        void setupTopics();
        void setupDefaultStreamCalibData();
        void publishStaticTransforms();
        void publishDynamicTransforms();
        void calcAndPublishStaticTransform();
        OBCameraParam findDefaultCameraParam();
        void publishStaticTF(const ros::Time& t, const std::vector<float>& trans,
                                   const tf2::Quaternion& q, const std::string& from,
                                   const std::string& to);
        OBCameraParam findStreamCameraParam(const stream_index_pair& stream,
                                                            uint32_t width, uint32_t height);

        OBCameraParam findCameraParam(uint32_t color_width, uint32_t color_height,
                                                    uint32_t depth_width, uint32_t depth_height);
        OBCameraParam findDepthCameraParam(uint32_t width, uint32_t height);

        OBCameraParam findColorCameraParam(uint32_t width, uint32_t height);

        void publishColorFrame(std::shared_ptr<ob::ColorFrame> frame);
        void publishDepthFrame(std::shared_ptr<ob::DepthFrame> frame);
        void publishIRFrame(std::shared_ptr<ob::IRFrame> frame);
        void publishPointCloud(std::shared_ptr<ob::FrameSet> frame_set);
        void publishDepthPointCloud(std::shared_ptr<ob::FrameSet> frame_set);
        void publishColorPointCloud(std::shared_ptr<ob::FrameSet> frame_set);
        bool getDepthCameraInfoCallback(orbbec_camera::GetCameraInfoRequest& req, orbbec_camera::GetCameraInfoResponse& res);
        bool getColorCameraInfoCallback(orbbec_camera::GetCameraInfoRequest& req, orbbec_camera::GetCameraInfoResponse& res);
        void getParameters();

        bool rbgFormatConvertRGB888(std::shared_ptr<ob::ColorFrame> frame);
};