#include "unsigned_distance_field.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_2.h>
#include <CGAL/AABB_segment_primitive_2.h>
#include <CGAL/squared_distance_2.h>

#include <list>
#include <iostream>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Segment_2 Segment_2;
typedef std::list<Segment_2> Segment2Range;
typedef std::list<Segment_2>::iterator Iterator;
typedef CGAL::AABB_segment_primitive_2<Kernel, Iterator> Primitive;
typedef CGAL::AABB_traits_2<Kernel, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef Tree::Point_and_primitive_id Point_and_primitive_id;


void unsigned_distance_field(
    const Eigen::MatrixXd& udf_points,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E,
    Eigen::VectorXd& udf_data
) {
    udf_data.resize(udf_points.rows());
    udf_data.setZero();

    // Convert the vertices (V) and edges (E) to CGAL data structures.
    std::list<Segment_2> edges;
    for (int i = 0; i < E.rows(); i ++) {
        Point_2 p1(V(E(i, 0), 0), V(E(i, 0), 1));
        Point_2 p2(V(E(i, 1), 0), V(E(i, 1), 1));
        edges.emplace_back(p1, p2);
    }
    // Construct the AABB tree and the internal search tree for
    //  efficient distance computations.
    Tree tree(edges.begin(), edges.end());
    tree.build();
    tree.accelerate_distance_queries();

    for (int i = 0; i < udf_points.rows(); i ++) {
        Point_2 query_point(udf_points(i, 0), udf_points(i, 1));
        double distance_squared = tree.squared_distance(query_point);
        udf_data(i) = std::sqrt(distance_squared);
    }
}
