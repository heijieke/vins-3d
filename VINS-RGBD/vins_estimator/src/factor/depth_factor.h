#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class DepthFactor : public ceres::SizedCostFunction<1, 7, 7, 7>
{
  public:
    DepthFactor(const Eigen::Vector3d &p_i, const Eigen::Vector3d &p_j, const double &weight);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    
    Eigen::Vector3d p_i,p_j;
    double weight;
};