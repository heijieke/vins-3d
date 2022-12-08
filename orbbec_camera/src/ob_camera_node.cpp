#include <ros/ros.h>
#include "libobsensor/ObSensor.hpp"
#include "OBCamera.h"

using namespace std;

int main(int argc, char **argv){
    ros::init(argc, argv, "sensorData_extractor");
    ros::NodeHandle nh;
    ob::Context ctx;

    //查询已经接入设备的列表
    auto devList = ctx.queryDeviceList();
    //获取接入设备的数量
    if(devList->deviceCount() == 0) {
        std::cerr << "Device not found!" << std::endl;
        return -1;
    }
    //创建设备，0表示第一个设备的索引
    auto dev = devList->getDevice(0);
    OBCamera ob(nh, dev);

    ros::spin();
    return 0;
};