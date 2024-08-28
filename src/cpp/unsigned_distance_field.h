#ifndef UNSIGNED_DISTANCE_FIELD_H 
#define UNSIGNED_DISTANCE_FIELD_H 

#include <Eigen/Core>

void unsigned_distance_field(
    const Eigen::MatrixXd& udf_points,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E,
    Eigen::VectorXd& udf_data
);

#endif