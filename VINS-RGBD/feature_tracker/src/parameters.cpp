#include "parameters.h"

std::string IMAGE_TOPIC;
std::string DEPTH_TOPIC;
std::string IMU_TOPIC;
std::string POINT_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;
double CUT_BEGIN, CUT_END;
double cx,cy,fx,fy;
int g_LiDAR_sampling_point_step,group_size;
double depth_scale;
double RESOLUTION;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    std::string depth_config_file = readParam<std::string>(n, "depth_config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    cv::FileStorage depthSettings(depth_config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened() && !depthSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["depth_topic"] >> DEPTH_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["point_topic"] >> POINT_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    CUT_BEGIN = fsSettings["cut_begin"];
    CUT_END = fsSettings["cut_end"];
    cx = fsSettings["projection_parameters"]["cx"];
    cy = fsSettings["projection_parameters"]["cy"];
    fx = fsSettings["projection_parameters"]["fx"];
    fy = fsSettings["projection_parameters"]["fy"];
    g_LiDAR_sampling_point_step = fsSettings["g_LiDAR_sampling_point_step"];
    group_size = fsSettings["group_size"];
    depth_scale = fsSettings["depth_scale"];
    RESOLUTION = fsSettings["Voxel_res1"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;//shan:What's this?---seems a virtual focal used in rejectWithF.
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
