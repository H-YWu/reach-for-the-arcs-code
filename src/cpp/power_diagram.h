#ifndef POWER_DIAGRAM_H 
#define POWER_DIAGRAM_H 

#include <Eigen/Core>

void power_diagram_and_crust_from_sdf(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::MatrixXd& sdf_data,
    Eigen::MatrixXd& V_power_diagram_out,
    Eigen::MatrixXd& V_power_diagram_in,
    Eigen::MatrixXd& V_power_crust,
    Eigen::MatrixXi& E_power_diagram_out,
    Eigen::MatrixXi& E_power_diagram_in,
    Eigen::MatrixXi& E_power_crust
);

#endif