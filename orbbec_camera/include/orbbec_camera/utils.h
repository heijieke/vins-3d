/**************************************************************************/
/*                                                                        */
/* Copyright (c) 2013-2021 Orbbec 3D Technology, Inc                      */
/*                                                                        */
/* PROPRIETARY RIGHTS of Orbbec 3D Technology are involved in the         */
/* subject matter of this material. All manufacturing, reproduction, use, */
/* and sales rights pertaining to this subject matter are governed by the */
/* license agreement. The recipient of this software implicitly accepts   */
/* the terms of the license.                                              */
/*                                                                        */
/**************************************************************************/

#pragma once
#include <ostream>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>
#include <ros/ros.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/distortion_models.h>

#include "libobsensor/ObSensor.hpp"
#include "orbbec_camera/Extrinsics.h"

namespace orbbec_camera {
sensor_msgs::CameraInfo convertToCameraInfo(OBCameraIntrinsic intrinsic,
                                                 OBCameraDistortion distortion);

void saveRGBPointsToPly(std::shared_ptr<ob::Frame> frame, std::string fileName);

void savePointsToPly(std::shared_ptr<ob::Frame> frame, std::string fileName);

tf2::Quaternion rotationMatrixToQuaternion(const float rotation[9]);

std::ostream& operator<<(std::ostream& os, const OBCameraParam& rhs);

orbbec_camera::Extrinsics obExtrinsicsToMsg(const OBD2CTransform& extrinsics,
                                                      const std::string& frame_id);

ros::Time frameTimeStampToROSTime(uint64_t ms);

std::string getObSDKVersion();

OBFormat OBFormatFromString(const std::string& format);

}  // namespace orbbec_camera
