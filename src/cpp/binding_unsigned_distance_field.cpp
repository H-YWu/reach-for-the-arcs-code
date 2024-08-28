#include "unsigned_distance_field.h"
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

void binding_unsigned_distance_field(py::module& m) {
    m.def("_unsigned_distance_field_cpp_impl",[](
            EigenDRef<MatrixXd> _udf_points,
            EigenDRef<MatrixXd> _V,
            EigenDRef<MatrixXi> _E)
        {
            Eigen::MatrixXd udf_points(_udf_points);
            const int dim = udf_points.cols();
            Eigen::MatrixXd V = _V;
            Eigen::MatrixXi E = _E;
            Eigen::VectorXd udf_data;
            if(dim==2) {
                unsigned_distance_field(udf_points, V, E, udf_data);
            } else if(dim==3) {
                // TODO 3D
            }
            return udf_data;
        });
}
