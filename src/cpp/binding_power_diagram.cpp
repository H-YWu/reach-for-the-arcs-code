#include "power_diagram.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
#include <iostream>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

void binding_power_diagram(py::module& m) {
    m.def("_power_diagram_cpp_impl",[](
            EigenDRef<MatrixXd> _sdf_points,
            EigenDRef<MatrixXd> _sdf_data)
        {
            Eigen::MatrixXd sdf_points(_sdf_points), sdf_data(_sdf_data);
            const int dim = sdf_points.cols();
            Eigen::MatrixXd V;
            Eigen::MatrixXi E;
            BoolVector Es;
            if(dim==2) {
                power_diagram_2d(sdf_points, sdf_data, V, E, Es);
            } else if(dim==3) {
                // TODO 3D
            }
            return std::make_tuple(V, E, Es);
        });
}
