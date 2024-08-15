#ifndef POWER_DIAGRAM_H 
#define POWER_DIAGRAM_H 

#include <Eigen/Core>

typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolVector;

void power_diagram_2d(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sdf_data,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & E,
    BoolVector & Es
);

#endif