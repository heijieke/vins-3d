#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class DepthFactor : public ceres::SizedCostFunction<3, 7, 7, 7>
{
  public:
    DepthFactor(const pcl::PointCloud<pcl::PointXYZ> &p_i,const pcl::PointCloud<pcl::PointXYZ> &p_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    pcl::PointCloud<pcl::PointXYZ> pl_i,pl_j;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_in;
};