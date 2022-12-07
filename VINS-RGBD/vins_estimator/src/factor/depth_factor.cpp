#include "depth_factor.h"

DepthFactor::DepthFactor(const pcl::PointCloud<pcl::PointXYZ> &p_i,const pcl::PointCloud<pcl::PointXYZ> &p_j) : pl_i(p_i), pl_j(p_j)
{
    p_in = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    *p_in = pl_j;
    TicToc tic_toc;
    kdtree.setInputCloud (p_in);
    std::cout << "kdtree cost：" << tic_toc.toc() << std::endl;
};

bool DepthFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const{
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);


    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translation() = tic;
    transform.rotate(qic);
    pcl::PointCloud<pcl::PointXYZ> pl_bi;
    pcl::PointCloud<pcl::PointXYZ> pl_wi;
    pcl::PointCloud<pcl::PointXYZ> pl_wj;
    pcl::PointCloud<pcl::PointXYZ> pl_bj;

    pcl::transformPointCloud(pl_i, pl_bi, transform);
    transform.translation() = Pi;
    transform.rotate(Qi);
    pcl::transformPointCloud(pl_bi, pl_wi, transform);
    transform.translation() = Qj.inverse() * (-Pj);
    transform.rotate(Qj.inverse());
    pcl::transformPointCloud(pl_wi, pl_wj, transform);
    transform.translation() = qic.inverse() * (-tic);
    transform.rotate(qic.inverse());
    pcl::transformPointCloud(pl_wj, pl_bj, transform);
    Eigen::Map<Eigen::Vector3d> residual(residuals);
    residual.setZero();
    
    double radius = 10; //单位mm
    std::vector<int> pointIdxSearch;
    std::vector<float> pointSquaredDistance;
    TicToc tic_toc;

    for(auto point : pl_bj){
        if(kdtree.nearestKSearch(point, 1, pointIdxSearch, pointSquaredDistance) > 0 && pointSquaredDistance[0] < radius){
            double d_x = (*p_in)[pointIdxSearch[0]].x - point.x;
            double d_y = (*p_in)[pointIdxSearch[0]].y - point.y;
            double d_z = (*p_in)[pointIdxSearch[0]].z - point.z;
            residual +=  Eigen::Vector3d(d_x * d_x, d_y * d_y, d_z * d_z);

        }
    }
    std::cout << "nearestKSearch cost：" << tic_toc.toc() << std::endl;


    return true;
}