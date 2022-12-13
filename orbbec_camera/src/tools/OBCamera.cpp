#include <OBCamera.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <geometry_msgs/TransformStamped.h>

OBCamera::OBCamera(ros::NodeHandle& nh,const std::shared_ptr<ob::Device> device):nh_(nh), device_(device){
  format_[DEPTH] = OB_FORMAT_Y16;
  format_str_[DEPTH] = "Y16";
  image_format_[OB_STREAM_DEPTH] = CV_16UC1;
  format_[INFRA0] = OB_FORMAT_Y16;
  format_str_[INFRA0] = "Y16";
  image_format_[OB_STREAM_IR] = CV_16UC1;
  format_[COLOR] = OB_FORMAT_I420;
  format_str_[COLOR] = "I420";
  image_format_[OB_STREAM_COLOR] = CV_8UC3;
  encoding_[DEPTH] = sensor_msgs::image_encodings::TYPE_16UC1;
  encoding_[INFRA0] = sensor_msgs::image_encodings::MONO16;
  encoding_[COLOR] = sensor_msgs::image_encodings::BGR8;

  stream_name_[OB_STREAM_COLOR] = "color";
  stream_name_[OB_STREAM_DEPTH] = "depth";
  stream_name_[OB_STREAM_IR] = "ir";

  unit_step_size_[COLOR] = 3;
  unit_step_size_[DEPTH] = sizeof(uint16_t);
  unit_step_size_[INFRA0] = sizeof(uint8_t);

  compression_params_.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params_.push_back(0);
  compression_params_.push_back(cv::IMWRITE_PNG_STRATEGY);
  compression_params_.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);
  is_running_.store(true);

  setupTopics();
  startPipeline();
}
OBCamera::~OBCamera(){clean();}
void OBCamera::clean(){
    is_running_.store(false);
    ROS_INFO("OBCamera disconection!");
    pipeline_->stop();
    ROS_INFO("stop camera");
    gyroSensor_->stop();
    ROS_INFO("stop gyro");
    accelSensor_->stop();
    ROS_INFO("stop accel");
}

void OBCamera::setupDevices() {
    auto sensor_list = device_->getSensorList();
    for (size_t i = 0; i < sensor_list->count(); i++) {
        auto sensor = sensor_list->getSensor(i);
        auto profiles = sensor->getStreamProfileList();
        for (size_t j = 0; j < profiles->count(); j++) {
          auto profile = profiles->getProfile(j);
          stream_index_pair sip{profile->type(), 0};
          if (sensors_.find(sip) != sensors_.end()) {
              continue;
          }
          sensors_[sip] = sensor;
        }
    }

    for (const auto& [stream_index, enable] : enable_) {
      if (enable && sensors_.find(stream_index) == sensors_.end()) {
        ROS_INFO_STREAM(stream_index.first << "sensor isn't supported by current device! -- Skipping...");
        enable_[stream_index] = false;
      }
  }
}
void OBCamera::setupProfiles() {
    if (config_ != nullptr) {
        config_.reset();
    }
    config_ = std::make_shared<ob::Config>();
    if (d2c_mode_ == "sw") {
        config_->setAlignMode(ALIGN_D2C_SW_MODE);
        align_depth_ = true;
    } else if (d2c_mode_ == "hw") {
        config_->setAlignMode(ALIGN_D2C_HW_MODE);
        align_depth_ = true;
    } else {
        config_->setAlignMode(ALIGN_DISABLE);
        align_depth_ = false;
    }

    for (const auto& elem : IMAGE_STREAMS) {
      if (enable_[elem]) {
        const auto& sensor = sensors_[elem];
        auto profiles = sensor->getStreamProfileList();
        for (size_t i = 0; i < profiles->count(); i++) {
          auto profile = profiles->getProfile(i)->as<ob::VideoStreamProfile>();
          ROS_DEBUG_STREAM("Sensor profile: " << "stream_type: " << profile->type()
                         << "Format: " << profile->format() << ", Width: " << profile->width()
                         << ", Height: " << profile->height() << ", FPS: " << profile->fps());
          enabled_profiles_[elem].emplace_back(profile);
        }

      auto selected_profile =
          profiles->getVideoStreamProfile(width_[elem], height_[elem], format_[elem], fps_[elem]);
      auto default_profile =
          profiles->getVideoStreamProfile(width_[elem], height_[elem], format_[elem]);
      if (!selected_profile) {
        ROS_WARN_STREAM("Given stream configuration is not supported by the device! "
                                        << " Stream: " << elem.first
                                        << ", Stream Index: " << elem.second
                                        << ", Width: " << width_[elem]
                                        << ", Height: " << height_[elem] << ", FPS: " << fps_[elem]
                                        << ", Format: " << format_[elem]);
        if (default_profile) {
          ROS_WARN_STREAM( "Using default profile instead." );
          ROS_WARN_STREAM("default FPS " << default_profile->fps() );
          selected_profile = default_profile;
        } else {
          ROS_ERROR_STREAM(" NO default_profile found , Stream: " << elem.first
                                                              << " will be disable");
          enable_[elem] = false;
          continue;
        }
      }
      CHECK_NOTNULL(selected_profile);
      config_->enableStream(selected_profile);
      images_[elem] =
          cv::Mat(height_[elem], width_[elem], image_format_[elem.first], cv::Scalar(0, 0, 0));
      ROS_INFO_STREAM(" stream " << stream_name_[elem.first] << " is enabled - width: " << width_[elem]
                              << ", height: " << height_[elem] << ", fps: " << fps_[elem] << ", "
                              << "Format: " << selected_profile->format());
    }
  }
}

void OBCamera::setupTopics() {
  getParameters();
  setupDevices();
  setupProfiles();
  setupDefaultStreamCalibData();
  setupPublishers();
  auto camera_param = findDefaultCameraParam();
  if (sizeof(camera_param)) {
    auto ex = camera_param.transform;
    extrinsics_publisher_.publish(orbbec_camera::obExtrinsicsToMsg(ex, "depth_to_color_extrinsics"));
  }
  //publishStaticTransforms();
  // depth_cameraInfo_ = orbbec_camera::convertToCameraInfo(camera_param.depthIntrinsic,camera_param.depthDistortion);
  // color_cameraInfo_ = orbbec_camera::convertToCameraInfo(camera_param.rgbIntrinsic,camera_param.rgbDistortion);
}


void OBCamera::setupPublishers() {
  using PointCloud2 = sensor_msgs::PointCloud2;
  using CameraInfo = sensor_msgs::CameraInfo;
  using ImuInfo = sensor_msgs::Imu;
  point_cloud_publisher_ = nh_.advertise<PointCloud2>("depth/color/points", 2000000);
  depth_point_cloud_publisher_ = nh_.advertise<PointCloud2>("depth/points", 2000000);
  imu_publisher_ = nh_.advertise<ImuInfo>("imu/imu", 2000000);
  image_transport::ImageTransport it(nh_);
  for (const auto& stream_index : IMAGE_STREAMS) {
    std::string name = stream_name_[stream_index.first];
    std::string topic = name + "/image_raw";
    image_publishers_[stream_index] = it.advertise(topic, 1000000);
    topic = name + "/camera_info";
    camera_info_publishers_[stream_index] = nh_.advertise<CameraInfo>(topic, 1);
  }
  extrinsics_publisher_ = nh_.advertise<orbbec_camera::Extrinsics>("extrinsic/depth_to_color", 1);
  imageG_publishers_ = it.advertise("gray/image_raw", 1000000);
}


void OBCamera::startPipeline() {
  accelSensor_ = device_->getSensorList()->getSensor(OB_SENSOR_ACCEL);
  gyroSensor_ = device_->getSensorList()->getSensor(OB_SENSOR_GYRO);
  if(gyroSensor_) {
    // 获取配置列表
    auto profiles = gyroSensor_->getStreamProfileList();
    // 选择第一个配置开流
    auto profile = profiles->getProfile(0);
    gyroSensor_->start(profile, [this](std::shared_ptr<ob::Frame> frame) {
      auto timestamp = orbbec_camera::frameTimeStampToROSTime(frame->systemTimeStamp());
      auto gyroFrame = frame->as<ob::GyroFrame>();
      // if(gyroFrame != nullptr && time/10 > last_time){
      //   auto gyro_value = gyroFrame->value();
      //   imu_msg_.header.stamp = timestamp;
      //   imu_msg_.header.frame_id = "imuFrame";
      //   imu_msg_.angular_velocity.x = gyro_value.x;
      //   imu_msg_.angular_velocity.y = gyro_value.y;
      //   imu_msg_.angular_velocity.z = gyro_value.z;
      // }
      if(gyroFrame != nullptr){
        auto gyro_value = gyroFrame->value();
        imu_msg_.header.stamp = timestamp;
        imu_msg_.header.frame_id = "imuFrame";
        imu_msg_.angular_velocity.x = gyro_value.x;
        imu_msg_.angular_velocity.y = gyro_value.y;
        imu_msg_.angular_velocity.z = gyro_value.z;
      }
    });
  }else {
        ROS_ERROR_STREAM("get gyro Sensor failed !/n");
  };

  if(accelSensor_) {
    // 获取配置列表
    auto profiles = accelSensor_->getStreamProfileList();
    // 选择第一个配置开流
    auto profile = profiles->getProfile(0);
    accelSensor_->start(profile, [this](std::shared_ptr<ob::Frame> frame) {
      auto accelFrame = frame->as<ob::AccelFrame>();
      // auto time = frame->timeStamp();
      // if(accelFrame != nullptr && time/10 > last_time){
      //   auto accel_value = accelFrame->value();
      //   imu_msg_.linear_acceleration.x = accel_value.x;
      //   imu_msg_.linear_acceleration.y = accel_value.y;
      //   imu_msg_.linear_acceleration.z = accel_value.z;
      //   imu_publisher_.publish(imu_msg_);
      //   last_time = time/10;
      //   if(last_time >= max_time)
      //     last_time = 0;
      // }

      if(accelFrame != nullptr){
        auto accel_value = accelFrame->value();
        imu_msg_.linear_acceleration.x = accel_value.x;
        imu_msg_.linear_acceleration.y = accel_value.y;
        imu_msg_.linear_acceleration.z = accel_value.z;
        imu_publisher_.publish(imu_msg_);
      }
    });
  }
  else {
      ROS_ERROR_STREAM( "get Accel Sensor failed !");
  }

  if (pipeline_ != nullptr) {
    pipeline_.reset();
  }
  pipeline_ = std::unique_ptr<ob::Pipeline>(new ob::Pipeline(device_));
  pipeline_->enableFrameSync();
  pipeline_->start(config_, [this](std::shared_ptr<ob::FrameSet> frame_set) {
    frameSetCallback(std::move(frame_set));
  });
}

void OBCamera::getParameters(){
  for (auto stream_index : IMAGE_STREAMS) {
    std::string param_name = "/camera__parameters/" + stream_name_[stream_index.first] + "_width";
    nh_.getParam(param_name,width_[stream_index]);
    param_name = "/camera__parameters/" + stream_name_[stream_index.first] + "_height";
    nh_.getParam(param_name,height_[stream_index]);
    param_name = "/camera__parameters/" + stream_name_[stream_index.first] + "_fps";
    nh_.getParam(param_name,fps_[stream_index]);
    param_name = "/camera__parameters/" + stream_name_[stream_index.first] + "_frame_id";
    nh_.getParam(param_name,frame_id_[stream_index]);
    param_name = "/camera__parameters/" + stream_name_[stream_index.first] + "_optical_frame_id";
    nh_.getParam(param_name,optical_frame_id_[stream_index]);
    param_name = "/camera__parameters/enable_" + stream_name_[stream_index.first];
    nh_.getParam(param_name,enable_[stream_index]);
    depth_aligned_frame_id_[stream_index] = stream_name_[OB_STREAM_COLOR] + "_optical_frame";
    format_[stream_index] = orbbec_camera::OBFormatFromString(format_str_[stream_index]);
  }
  nh_.getParam("/camera__parameters/publish_tf",publish_tf_);
  nh_.getParam("/camera__parameters/tf_publish_rate",tf_publish_rate_);
  nh_.getParam("/camera__parameters/publish_rgb_point_cloud",publish_rgb_point_cloud_);
  nh_.getParam("/camera__parameters/d2c_mode",d2c_mode_);
  nh_.getParam("/camera__parameters/camera_link_frame_id", camera_link_frame_id_);
}

void OBCamera::frameSetCallback(std::shared_ptr<ob::FrameSet> frame_set) {
    // 只发布需要用到的点云、深度图和彩色图，其他的不发布节省资源
    auto color_frame = frame_set->colorFrame();
    auto depth_frame = frame_set->depthFrame();
    auto ir_frame = frame_set->irFrame();
    if (color_frame && enable_[COLOR]) {
      publishColorFrame(color_frame);
    }
    if (depth_frame && enable_[DEPTH]) {
      publishDepthFrame(depth_frame);
    }
    if (ir_frame && enable_[INFRA0]) {
      publishIRFrame(ir_frame);
    }
    publishPointCloud(frame_set);
}


OBCameraParam OBCamera::findDefaultCameraParam() {
  auto camera_params = device_->getCalibrationCameraParamList();
  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int depth_w = param.depthIntrinsic.width;
    int depth_h = param.depthIntrinsic.height;
    int color_w = param.rgbIntrinsic.width;
    int color_h = param.rgbIntrinsic.height;
    if ((depth_w * depth_height_ == depth_h * depth_width_) &&
        (color_w * color_height_ == color_h * color_width_)) {
      return param;
    }
  }
  return {};
}

OBCameraParam OBCamera::findCameraParam(uint32_t color_width,
                                                           uint32_t color_height,
                                                           uint32_t depth_width,
                                                           uint32_t depth_height) {
  auto camera_params = device_->getCalibrationCameraParamList();
  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int depth_w = param.depthIntrinsic.width;
    int depth_h = param.depthIntrinsic.height;
    int color_w = param.rgbIntrinsic.width;
    int color_h = param.rgbIntrinsic.height;
    if ((depth_w * depth_height == depth_h * depth_width) &&
        (color_w * color_height == color_h * color_width)) {
      return param;
    }
  }
  return {};
}

OBCameraParam OBCamera::findDepthCameraParam(uint32_t width, uint32_t height) {
  auto camera_params = device_->getCalibrationCameraParamList();
  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int depth_w = param.depthIntrinsic.width;
    int depth_h = param.depthIntrinsic.height;
    if (depth_w * height == depth_h * width) {
      return param;
    }
  }
  return {};
}

OBCameraParam OBCamera::findColorCameraParam(uint32_t width, uint32_t height) {
  auto camera_params = device_->getCalibrationCameraParamList();
  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int color_w = param.rgbIntrinsic.width;
    int color_h = param.rgbIntrinsic.height;
    if (color_w * height == color_h * width) {
      return param;
    }
  }
  return {};
}
void OBCamera::setupDefaultStreamCalibData() {
  auto param = findDefaultCameraParam();
  if (!sizeof(param)) {
    ROS_ERROR_STREAM("Not Found default camera parameter");
    align_depth_ = false;
    return;
  } else {
    updateStreamCalibData(param);
  }
}

void OBCamera::updateStreamCalibData(const OBCameraParam& param) {
  camera_infos_[DEPTH] = orbbec_camera::convertToCameraInfo(param.depthIntrinsic, param.depthDistortion);
  camera_infos_[COLOR] = orbbec_camera::convertToCameraInfo(param.rgbIntrinsic, param.rgbDistortion);
  camera_infos_[INFRA0] = camera_infos_[DEPTH];
}

void OBCamera::publishStaticTF(const ros::Time& t, const std::vector<float>& trans,
                                   const tf2::Quaternion& q, const std::string& from,
                                   const std::string& to) {
  CHECK_EQ(trans.size(), 3u);
  geometry_msgs::TransformStamped msg;
  msg.header.stamp = t;
  msg.header.seq = 10;
  msg.header.frame_id = from;
  msg.child_frame_id = to;
  msg.transform.translation.x = trans.at(2) / 1000.0;
  msg.transform.translation.y = -trans.at(0) / 1000.0;
  msg.transform.translation.z = -trans.at(1) / 1000.0;
  msg.transform.rotation.x = q.getX();
  msg.transform.rotation.y = q.getY();
  msg.transform.rotation.z = q.getZ();
  msg.transform.rotation.w = q.getW();
  static_tf_msgs_.push_back(msg);
}

void OBCamera::calcAndPublishStaticTransform() {
  tf2::Quaternion quaternion_optical, zero_rot, Q;
  std::vector<float> trans(3, 0);
  zero_rot.setRPY(0.0, 0.0, 0.0);
  quaternion_optical.setRPY(-M_PI / 2, 0.0, -M_PI / 2);
  std::vector<float> zero_trans = {0, 0, 0};
  auto camera_param = findDefaultCameraParam();
  if (sizeof(camera_param)) {
    auto ex = camera_param.transform;
    Q = orbbec_camera::rotationMatrixToQuaternion(ex.rot);
    Q = quaternion_optical * Q * quaternion_optical.inverse();
    extrinsics_publisher_.publish(orbbec_camera::obExtrinsicsToMsg(ex, "depth_to_color_extrinsics"));
  } else {
    Q.setRPY(0, 0, 0);
  }
  ros::Time tf_timestamp = ros::Time::now();

  publishStaticTF(tf_timestamp, trans, Q, frame_id_[DEPTH], frame_id_[COLOR]);
  publishStaticTF(tf_timestamp, trans, Q, camera_link_frame_id_, frame_id_[COLOR]);
  publishStaticTF(tf_timestamp, zero_trans, quaternion_optical, frame_id_[COLOR],
                  optical_frame_id_[COLOR]);
  publishStaticTF(tf_timestamp, zero_trans, quaternion_optical, frame_id_[DEPTH],
                  optical_frame_id_[DEPTH]);
  publishStaticTF(tf_timestamp, zero_trans, zero_rot, camera_link_frame_id_, frame_id_[DEPTH]);
}

void OBCamera::publishStaticTransforms() {
  calcAndPublishStaticTransform();
  if (tf_publish_rate_ > 0) {
    tf_thread_ = std::make_shared<std::thread>([this]() { publishDynamicTransforms(); });
  } else {
    static_tf_broadcaster_->sendTransform(static_tf_msgs_);
  }
}

void OBCamera::publishDynamicTransforms() {
  ROS_WARN("Publishing dynamic camera transforms (/tf) at %g Hz", tf_publish_rate_);
  std::mutex mu;
  std::unique_lock<std::mutex> lock(mu);
  while (ros::ok() && is_running_) {
    tf_cv_.wait_for(lock, std::chrono::milliseconds((int)(1000.0 / tf_publish_rate_)),
                    [this] { return (!(is_running_)); });
    {
      ros::Time t = ros::Time::now();
      for (auto& msg : static_tf_msgs_) {
        msg.header.stamp = t;
      }
      //dynamic_tf_broadcaster_->sendTransform(static_tf_msgs_);
      for(int i = 0;i < static_tf_msgs_.size();i++){
        static_tf_broadcaster_->sendTransform(static_tf_msgs_[i]);
      }
    }
  }
}

void OBCamera::publishColorFrame(std::shared_ptr<ob::ColorFrame> frame) {
  if (!rbgFormatConvertRGB888(frame)) {
    ROS_ERROR_STREAM("can not convert " << frame->format() << " to RGB888");
    return;
  }
  frame = format_convert_filter_.process(frame)->as<ob::ColorFrame>();
  format_convert_filter_.setFormatConvertType(FORMAT_RGB888_TO_BGR);
  frame = format_convert_filter_.process(frame)->as<ob::ColorFrame>();
  auto width = frame->width();
  auto height = frame->height();
  auto stream = COLOR;
  auto& image = images_[stream];
  if (image.size() != cv::Size(width, height)) {
    image.create(height, width, image.type());
  }
  image.data = (uint8_t*)frame->data();
  auto timestamp = orbbec_camera::frameTimeStampToROSTime(frame->systemTimeStamp());
  if (camera_infos_.count(stream)) {
    auto& cam_info = camera_infos_.at(stream);
    if (cam_info.width != width || cam_info.height != height) {
      updateStreamCalibData(pipeline_->getCameraParam());
      cam_info.height = height;
      cam_info.width = width;
    }
    cam_info.header.stamp = timestamp;
    auto& camera_info_publisher = camera_info_publishers_.at(stream);
    camera_info_publisher.publish(cam_info);
  }
  sensor_msgs::ImagePtr img;
  img = cv_bridge::CvImage(std_msgs::Header(), encoding_.at(stream), image).toImageMsg();
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  sensor_msgs::ImagePtr img_gray = cv_bridge::CvImage(std_msgs::Header(), "mono8", image_gray).toImageMsg();

  img_gray->width = width;
  img_gray->height = height;
  img_gray->is_bigendian = false;
  img_gray->step = width * sizeof(uint8_t);
  img_gray->header.frame_id = optical_frame_id_[COLOR];
  img_gray->header.stamp = timestamp;

  img->width = width;
  img->height = height;
  img->is_bigendian = false;
  img->step = width * unit_step_size_[stream];
  img->header.frame_id = optical_frame_id_[COLOR];
  img->header.stamp = timestamp;
  auto& image_publisher = image_publishers_.at(stream);
  image_publisher.publish(img);
  imageG_publishers_.publish(img_gray);
}

void OBCamera::publishDepthFrame(std::shared_ptr<ob::DepthFrame> frame) {
  auto width = frame->width();
  auto height = frame->height();
  auto stream = DEPTH;
  auto& image = images_[stream];
  if (image.size() != cv::Size(width, height)) {
    image.create(height, width, image.type());
  }
  image.data = (uint8_t*)frame->data();
  auto timestamp = orbbec_camera::frameTimeStampToROSTime(frame->systemTimeStamp());
  if (camera_infos_.count(stream)) {
    auto& cam_info = camera_infos_.at(stream);
    if (cam_info.width != width || cam_info.height != height) {
      updateStreamCalibData(pipeline_->getCameraParam());
      cam_info.height = height;
      cam_info.width = width;
    }
    cam_info.header.stamp = timestamp;
    auto& camera_info_publisher = camera_info_publishers_.at(stream);
    camera_info_publisher.publish(cam_info);
  }
  sensor_msgs::ImagePtr img;
  img = cv_bridge::CvImage(std_msgs::Header(), encoding_.at(stream), image).toImageMsg();

  img->width = width;
  img->height = height;
  img->is_bigendian = false;
  img->step = width * unit_step_size_[stream];
  if (align_depth_) {
    img->header.frame_id = depth_aligned_frame_id_[DEPTH];
  } else {
    img->header.frame_id = optical_frame_id_[DEPTH];
  }
  img->header.stamp = timestamp;
  auto& image_publisher = image_publishers_.at(stream);
  image_publisher.publish(img);
}

void OBCamera::publishIRFrame(std::shared_ptr<ob::IRFrame> frame) {
  auto width = frame->width();
  auto height = frame->height();
  auto stream = INFRA0;
  auto& image = images_[stream];
  if (image.size() != cv::Size(width, height)) {
    image.create(height, width, image.type());
  }
  image.data = (uint8_t*)frame->data();
  auto timestamp = orbbec_camera::frameTimeStampToROSTime(frame->systemTimeStamp());
  auto& image_publisher = image_publishers_.at(stream);
  if (camera_infos_.count(stream)) {
    auto& cam_info = camera_infos_.at(stream);
    auto& camera_info_publisher = camera_info_publishers_.at(stream);

    if (cam_info.width != width || cam_info.height != height) {
      updateStreamCalibData(pipeline_->getCameraParam());
      cam_info.height = height;
      cam_info.width = width;
    }
    cam_info.header.stamp = timestamp;
    camera_info_publisher.publish(cam_info);
  }
  sensor_msgs::ImagePtr img;
  img = cv_bridge::CvImage(std_msgs::Header(), encoding_.at(stream), image).toImageMsg();

  img->width = width;
  img->height = height;
  img->is_bigendian = false;
  img->step = width * unit_step_size_[stream];
  if (align_depth_) {
    img->header.frame_id = depth_aligned_frame_id_[DEPTH];
  } else {
    img->header.frame_id = optical_frame_id_[DEPTH];
  }
  img->header.stamp = timestamp;
  image_publisher.publish(img);
}


void OBCamera::publishPointCloud(std::shared_ptr<ob::FrameSet> frame_set) {
  try {
    if (frame_set->depthFrame() != nullptr && frame_set->colorFrame() != nullptr) {
      publishColorPointCloud(frame_set);
    }
    if (frame_set->depthFrame() != nullptr) {
      publishDepthPointCloud(frame_set);
    }
  } catch (const ob::Error& e) {
    ROS_ERROR_STREAM(e.getMessage());
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM(e.what());
  } catch (...) {
    ROS_ERROR_STREAM("publishPointCloud with unknown error");
  }
}

// void OBCamera::publishDepthPointCloud(std::shared_ptr<ob::FrameSet> frame_set) {
//   if (depth_point_cloud_publisher_.getNumSubscribers() == 0) {
//     return;
//   }
//   auto camera_param = pipeline_->getCameraParam();
//   point_cloud_filter_.setCameraParam(camera_param);
//   point_cloud_filter_.setCreatePointFormat(OB_FORMAT_POINT);
//   auto depth_frame = frame_set->depthFrame();
//   auto frame = point_cloud_filter_.process(frame_set);
//   size_t point_size = frame->dataSize() / sizeof(OBPoint);
//   auto* points = (OBPoint*)frame->data();
//   CHECK_NOTNULL(points);
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr input(new pcl::PointCloud<pcl::PointXYZRGB>);
//   for(size_t point_idx = 0; point_idx < point_size; point_idx++){
//     if((points + point_idx)->z <= 0)
//       continue;
//     pcl::PointXYZRGB p;
//     p.x = static_cast<float>((points + point_idx)->x / 1000.0);
//     p.y = static_cast<float>((points + point_idx)->y / 1000.0);
//     p.z = static_cast<float>((points + point_idx)->z / 1000.0);
//     p.r = 0;
//     p.g = 0;
//     p.b = 0;
//     input->points.push_back(p);
//   }
//   // pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
//   // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statistical_filter;
//   // statistical_filter.setMeanK(50);
//   // statistical_filter.setStddevMulThresh(1.0);
//   // statistical_filter.setInputCloud(input);
//   // statistical_filter.filter(*tmp);

//  //下采样，不然密度太高了
//   pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
//   voxel_filter.setLeafSize(0.01, 0.01, 0.01);
//   voxel_filter.setInputCloud(input);
//   voxel_filter.filter(*output);

//   //计算曲率
//   // pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//   // ne.setInputCloud(output);
//   // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//   // ne.setSearchMethod(tree);
//   // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//   // ne.setRadiusSearch(0.002);
//   // ne.compute(*cloud_normals);

//   // pcl::PointCloud<pcl::PointXYZINormal>::Ptr res(new pcl::PointCloud<pcl::PointXYZINormal>);
//   // for(int i = 0; i < output->points.size(); i++){
//   //   pcl::PointXYZINormal p;
//   //   p.x = output->points[i].x;
//   //   p.y = output->points[i].y;
//   //   p.z = output->points[i].z;
//   //   p.normal_x = cloud_normals->points[i].normal_x;
//   //   p.normal_y = cloud_normals->points[i].normal_y;
//   //   p.normal_z = cloud_normals->points[i].normal_z;
//   //   p.curvature = cloud_normals->points[i].curvature;
//   //   p.intensity = 0;
//   //   res->points.push_back(p);
//   // }

//   pcl::toROSMsg(*output, point_cloud_msg_);
//   auto timestamp = orbbec_camera::frameTimeStampToROSTime(depth_frame->systemTimeStamp());
//   point_cloud_msg_.header.stamp = timestamp;
//   //point_cloud_msg_.header.frame_id = optical_frame_id_[DEPTH];
//   point_cloud_msg_.header.frame_id = "color_point_cloud";
//   point_cloud_msg_.is_dense = true;
//   point_cloud_publisher_.publish(point_cloud_msg_);
// }

void OBCamera::publishDepthPointCloud(std::shared_ptr<ob::FrameSet> frame_set) {
  if (depth_point_cloud_publisher_.getNumSubscribers() == 0) {
    return;
  }
  auto camera_param = pipeline_->getCameraParam();
  point_cloud_filter_.setCameraParam(camera_param);
  point_cloud_filter_.setCreatePointFormat(OB_FORMAT_POINT);
  auto depth_frame = frame_set->depthFrame();
  auto frame = point_cloud_filter_.process(frame_set);
  size_t point_size = frame->dataSize() / sizeof(OBColorPoint);
  auto* points = (OBColorPoint*)frame->data();
  CHECK_NOTNULL(points);
  sensor_msgs::PointCloud2Modifier modifier(point_cloud_msg_);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(point_size);
  point_cloud_msg_.width = depth_frame->width();
  point_cloud_msg_.height = depth_frame->height();
  std::string format_str = "rgb";
  point_cloud_msg_.point_step =
      addPointField(point_cloud_msg_, format_str.c_str(), 1, sensor_msgs::PointField::FLOAT32,
                    point_cloud_msg_.point_step);
  point_cloud_msg_.row_step = point_cloud_msg_.width * point_cloud_msg_.point_step;
  point_cloud_msg_.data.resize(point_cloud_msg_.height * point_cloud_msg_.row_step);
  sensor_msgs::PointCloud2Iterator<float> iter_x(point_cloud_msg_, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(point_cloud_msg_, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(point_cloud_msg_, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(point_cloud_msg_, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(point_cloud_msg_, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(point_cloud_msg_, "b");
  size_t valid_count = 0;
  for (size_t point_idx = 0; point_idx < point_size; point_idx += 1) {
    bool valid_pixel((points + point_idx)->z > 0);
    if (valid_pixel) {
      *iter_x = static_cast<float>((points + point_idx)->x / 1000.0);
      *iter_y = -static_cast<float>((points + point_idx)->y / 1000.0);
      *iter_z = static_cast<float>((points + point_idx)->z / 1000.0);
      *iter_r = 0;
      *iter_g = 0;
      *iter_b = 0;

      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_r;
      ++iter_g;
      ++iter_b;
      ++valid_count;
    }
  }

  auto timestamp = orbbec_camera::frameTimeStampToROSTime(depth_frame->systemTimeStamp());
  point_cloud_msg_.header.stamp = timestamp;
  point_cloud_msg_.header.frame_id = "color_point_cloud";//optical_frame_id_[COLOR];
  point_cloud_msg_.is_dense = true;
  point_cloud_msg_.width = valid_count;
  point_cloud_msg_.height = 1;

  modifier.resize(valid_count);
  point_cloud_publisher_.publish(point_cloud_msg_);
}

void OBCamera::publishColorPointCloud(std::shared_ptr<ob::FrameSet> frame_set) {
  if (point_cloud_publisher_.getNumSubscribers() == 0) {
    return;
  }
  auto depth_frame = frame_set->depthFrame();
  auto color_frame = frame_set->colorFrame();
  auto camera_param = pipeline_->getCameraParam();
  point_cloud_filter_.setCameraParam(camera_param);
  point_cloud_filter_.setCreatePointFormat(OB_FORMAT_RGB_POINT);
  auto frame = point_cloud_filter_.process(frame_set);
  size_t point_size = frame->dataSize() / sizeof(OBColorPoint);
  auto* points = (OBColorPoint*)frame->data();
  CHECK_NOTNULL(points);
  sensor_msgs::PointCloud2Modifier modifier(point_cloud_msg_);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(point_size);
  point_cloud_msg_.width = color_frame->width();
  point_cloud_msg_.height = color_frame->height();
  std::string format_str = "rgb";
  point_cloud_msg_.point_step =
      addPointField(point_cloud_msg_, format_str.c_str(), 1, sensor_msgs::PointField::FLOAT32,
                    point_cloud_msg_.point_step);
  point_cloud_msg_.row_step = point_cloud_msg_.width * point_cloud_msg_.point_step;
  point_cloud_msg_.data.resize(point_cloud_msg_.height * point_cloud_msg_.row_step);
  sensor_msgs::PointCloud2Iterator<float> iter_x(point_cloud_msg_, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(point_cloud_msg_, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(point_cloud_msg_, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(point_cloud_msg_, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(point_cloud_msg_, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(point_cloud_msg_, "b");
  size_t valid_count = 0;
  for (size_t point_idx = 0; point_idx < point_size; point_idx += 1) {
    bool valid_pixel((points + point_idx)->z > 0);
    if (valid_pixel) {
      *iter_x = static_cast<float>((points + point_idx)->x / 1000.0);
      *iter_y = -static_cast<float>((points + point_idx)->y / 1000.0);
      *iter_z = static_cast<float>((points + point_idx)->z / 1000.0);
      *iter_r = static_cast<uint8_t>((points + point_idx)->r);
      *iter_g = static_cast<uint8_t>((points + point_idx)->g);
      *iter_b = static_cast<uint8_t>((points + point_idx)->b);

      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_r;
      ++iter_g;
      ++iter_b;
      ++valid_count;
    }
  }

  auto timestamp = orbbec_camera::frameTimeStampToROSTime(depth_frame->systemTimeStamp());
  point_cloud_msg_.header.stamp = timestamp;
  point_cloud_msg_.header.frame_id = "color_point_cloud";//optical_frame_id_[COLOR];
  point_cloud_msg_.is_dense = true;
  point_cloud_msg_.width = valid_count;
  point_cloud_msg_.height = 1;

  modifier.resize(valid_count);
  point_cloud_publisher_.publish(point_cloud_msg_);
}

// void OBCamera::publishColorPointCloud(std::shared_ptr<ob::FrameSet> frame_set) {
//   if (point_cloud_publisher_.getNumSubscribers() == 0) {
//     return;
//   }
//   auto depth_frame = frame_set->depthFrame();
//   auto color_frame = frame_set->colorFrame();
//   auto camera_param = pipeline_->getCameraParam();
//   point_cloud_filter_.setCameraParam(camera_param);
//   point_cloud_filter_.setCreatePointFormat(OB_FORMAT_RGB_POINT);
//   auto frame = point_cloud_filter_.process(frame_set);
//   size_t point_size = frame->dataSize() / sizeof(OBColorPoint);
//   auto* points = (OBColorPoint*)frame->data();
//   CHECK_NOTNULL(points);
  

//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr input(new pcl::PointCloud<pcl::PointXYZRGB>);
//   for(size_t point_idx = 0; point_idx < point_size; point_idx++){
//     if((points + point_idx)->z <= 0)
//       continue;
//     pcl::PointXYZRGB p;
//     p.x = static_cast<float>((points + point_idx)->x / 1000.0);
//     p.y = static_cast<float>((points + point_idx)->y / 1000.0);
//     p.z = static_cast<float>((points + point_idx)->z / 1000.0);
//     p.r = static_cast<uint8_t>((points + point_idx)->r);
//     p.g = static_cast<uint8_t>((points + point_idx)->g);
//     p.b = static_cast<uint8_t>((points + point_idx)->b);
//     input->points.push_back(p);
//   }
//   // pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
//   // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statistical_filter;
//   // statistical_filter.setMeanK(50);
//   // statistical_filter.setStddevMulThresh(1.0);
//   // statistical_filter.setInputCloud(input);
//   // statistical_filter.filter(*tmp);

//   pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
//   voxel_filter.setLeafSize(0.01, 0.01, 0.01);
//   voxel_filter.setInputCloud(input);
//   voxel_filter.filter(*output);

//   //计算曲率
//   // pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
//   // ne.setInputCloud(input);
//   // pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
//   // ne.setSearchMethod(tree);
//   // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//   // ne.setRadiusSearch(0.002);
//   // ne.compute(*cloud_normals);

//   // pcl::PointCloud<pcl::PointXYZRGB>::Ptr res(new pcl::PointCloud<pcl::PointXYZRGB>);
//   // for(int i = 0; i < input->points.size(); i++){
//   //   pcl::PointXYZRGB p;
//   //   p.x = input->points[i].x;
//   //   p.y = input->points[i].y;
//   //   p.z = input->points[i].z;
//   //   p.r = input->points[i].r;
//   //   p.g = input->points[i].g;
//   //   p.b = input->points[i].b;
//   //   p.normal_x = cloud_normals->points[i].normal_x;
//   //   p.normal_y = cloud_normals->points[i].normal_y;
//   //   p.normal_z = cloud_normals->points[i].normal_z;
//   //   p.curvature = cloud_normals->points[i].curvature;
//   //   p.intensity = (input->points[i].r + input->points[i].g + input->points[i].b)/3.0;
//   //   res->points.push_back(p);
//   // }

//   pcl::toROSMsg(*output, point_cloud_msg_);

//   auto timestamp = orbbec_camera::frameTimeStampToROSTime(depth_frame->systemTimeStamp());
//   point_cloud_msg_.header.stamp = timestamp;
//   point_cloud_msg_.header.frame_id = "color_point_cloud";//optical_frame_id_[COLOR];
//   point_cloud_publisher_.publish(point_cloud_msg_);
// }

bool OBCamera::rbgFormatConvertRGB888(std::shared_ptr<ob::ColorFrame> frame) {
  switch (frame->format()) {
    case OB_FORMAT_RGB888:
      return true;
    case OB_FORMAT_I420:
      format_convert_filter_.setFormatConvertType(FORMAT_I420_TO_RGB888);
      break;
    case OB_FORMAT_MJPG:
      format_convert_filter_.setFormatConvertType(FORMAT_MJPEG_TO_RGB888);
      break;
    case OB_FORMAT_YUYV:
      format_convert_filter_.setFormatConvertType(FORMAT_YUYV_TO_RGB888);
      break;
    case OB_FORMAT_NV21:
      format_convert_filter_.setFormatConvertType(FORMAT_NV21_TO_RGB888);
      break;
    case OB_FORMAT_NV12:
      format_convert_filter_.setFormatConvertType(FORMAT_NV12_TO_RGB888);
      break;
    default:
      return false;
  }
  return true;
}