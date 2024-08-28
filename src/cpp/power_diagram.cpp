#include "power_diagram.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>

#include <map>
#include <iterator>
#include <stdexcept>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Regular_triangulation_2<Kernel> Regular_triangulation_2;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Weighted_point_2 Weighted_point_2;
typedef Kernel::Iso_rectangle_2 Iso_rectangle_2;
typedef Kernel::Segment_2 Segment_2;
typedef Kernel::Ray_2 Ray_2;
typedef Kernel::Line_2 Line_2;


struct Cropped_power_diagram_from_regular_triangulation {
public:
    std::vector<Point_2> m_pd_out_verts;
    std::vector<Point_2> m_pd_in_verts;
    std::vector<Point_2> m_pc_verts;
    std::vector<std::pair<size_t, size_t>> m_pd_out_edges;
    std::vector<std::pair<size_t, size_t>> m_pd_in_edges;
    std::vector<std::pair<size_t, size_t>> m_pc_edges;

    Iso_rectangle_2 m_bbox;
    std::unordered_map<Point_2, size_t> m_pd_out_pt_indices;
    std::unordered_map<Point_2, size_t> m_pd_in_pt_indices;
    std::unordered_map<Point_2, size_t> m_pc_pt_indices;

    enum EdgeType {POWER_CRUST, POWER_DIAGRAM_OUT, POWER_DIAGRAM_IN};

    size_t index_of_point(std::unordered_map<Point_2, size_t>& index_map, std::vector<Point_2>& verts, const Point_2& p) {
        auto it = index_map.find(p);
        if (it == index_map.end()) {
            size_t idx = verts.size();
            verts.emplace_back(p);
            index_map[p] = idx;
            return idx;
        }
        return it->second;
    };

    Cropped_power_diagram_from_regular_triangulation(
        const Iso_rectangle_2& bbox,
        const Regular_triangulation_2& rt2,
        const std::unordered_map<Point_2, int>& psigns
    ) : m_bbox(bbox) {

        auto get_sign = [&](const Point_2& p) -> int {
            auto it = psigns.find(p);
            if (it == psigns.end()) {
                throw std::runtime_error("No sign assigned to this point!");
            }
            return it->second;
        };


        for (const Regular_triangulation_2::Edge& e : rt2.finite_edges()) {
            // Classify the edge 
            auto vh1 = e.first->vertex((e.second + 1) % 3);
            auto vh2 = e.first->vertex((e.second + 2) % 3);
            Point_2 c1 = vh1->point().point();
            Point_2 c2 = vh2->point().point();
            int s1 = get_sign(c1), s2 = get_sign(c2);
            EdgeType etype;
            if (s1 * s2 == -1) etype = POWER_CRUST; 
            else if (s1 == 1) etype = POWER_DIAGRAM_OUT;
            else etype = POWER_DIAGRAM_IN;
            CGAL::Object de = rt2.dual(e);
            // Extract a cropped power diagram edge
            if (const Segment_2* seg = CGAL::object_cast<Segment_2>(&de)) {
                crop_segment_and_add_edge(*seg, etype);
            } else if (const Ray_2* ray = CGAL::object_cast<Ray_2>(&de)) {
                crop_segment_and_add_edge(*ray, etype);
            } else if (const Line_2* line = CGAL::object_cast<Line_2>(&de)) {
                crop_segment_and_add_edge(*line, etype);
            } else {
                throw std::runtime_error("The dual of this edge is not a segment, ray or line!");
            }
        }
    }

    template <class RSL>
    void crop_segment_and_add_edge(const RSL& rsl, EdgeType etype) {
        // Intersect with the bounding box so that it becomes a segment
        CGAL::Object obj = CGAL::intersection(rsl, m_bbox);
        const Segment_2* s = CGAL::object_cast<Segment_2>(&obj);
        if (s) {
            Point_2 p1 = s->source();
            Point_2 p2 = s->target();
            if (etype == POWER_CRUST) {
                size_t idx1 = index_of_point(m_pc_pt_indices, m_pc_verts, p1);
                size_t idx2 = index_of_point(m_pc_pt_indices, m_pc_verts, p2);
                m_pc_edges.emplace_back(idx1, idx2);
            } else if (etype == POWER_DIAGRAM_OUT) {
                size_t idx1 = index_of_point(m_pd_out_pt_indices, m_pd_out_verts, p1);
                size_t idx2 = index_of_point(m_pd_out_pt_indices, m_pd_out_verts, p2);
                m_pd_out_edges.emplace_back(idx1, idx2);
            } else {
                size_t idx1 = index_of_point(m_pd_in_pt_indices, m_pd_in_verts, p1);
                size_t idx2 = index_of_point(m_pd_in_pt_indices, m_pd_in_verts, p2);
                m_pd_in_edges.emplace_back(idx1, idx2);
            }
        }
    }
};

void power_diagram_and_crust_from_sdf(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::MatrixXd& sdf_data,
    Eigen::MatrixXd& V_power_diagram_out,
    Eigen::MatrixXd& V_power_diagram_in,
    Eigen::MatrixXd& V_power_crust,
    Eigen::MatrixXi& E_power_diagram_out,
    Eigen::MatrixXi& E_power_diagram_in,
    Eigen::MatrixXi& E_power_crust
) {
    const int n = sdf_points.rows();
    assert(n == sdf_data.rows());

    // Build weighted points and sign map
    Eigen::ArrayXd signs = sdf_data.array().sign();
    Eigen::ArrayXd abs_radii = sdf_data.array().abs();
    std::vector<Weighted_point_2> wpoints;
    std::unordered_map<Point_2, int> psigns;
    for (int i = 0; i < n; i ++) {
        auto p = Point_2(sdf_points(i, 0), sdf_points(i, 1));
        wpoints.push_back(Weighted_point_2(p, abs_radii(i)*abs_radii(i)));
        psigns[p] = signs(i);
    }

    // Build bounding box
    Eigen::VectorXd x_coords = sdf_points.col(0);
    Eigen::VectorXd y_coords = sdf_points.col(1);
    double min_x = x_coords.minCoeff();
    double max_x = x_coords.maxCoeff();
    double min_y = y_coords.minCoeff();
    double max_y = y_coords.maxCoeff();
    Iso_rectangle_2 bbox(min_x, min_y, max_x, max_y);

    // Build cropped power diagram
    Regular_triangulation_2 rt2(wpoints.begin(), wpoints.end());
    Cropped_power_diagram_from_regular_triangulation cpd(bbox, rt2, psigns);

    // Convert std::vector to Eigen::Matrix
    V_power_diagram_out.resize(cpd.m_pd_out_verts.size(), 2);
    V_power_diagram_in.resize(cpd.m_pd_in_verts.size(), 2);
    V_power_crust.resize(cpd.m_pc_verts.size(), 2);
    E_power_diagram_out.resize(cpd.m_pd_out_edges.size(), 2);
    E_power_diagram_in.resize(cpd.m_pd_in_edges.size(), 2);
    E_power_crust.resize(cpd.m_pc_edges.size(), 2);
    for (size_t i = 0; i < cpd.m_pd_out_verts.size(); i ++) {
        Point_2 p = cpd.m_pd_out_verts[i];
        V_power_diagram_out(i, 0) = CGAL::to_double(p.x());
        V_power_diagram_out(i, 1) = CGAL::to_double(p.y());
    }
    for (size_t i = 0; i < cpd.m_pd_in_verts.size(); i ++) {
        Point_2 p = cpd.m_pd_in_verts[i];
        V_power_diagram_in(i, 0) = CGAL::to_double(p.x());
        V_power_diagram_in(i, 1) = CGAL::to_double(p.y());
    }
    for (size_t i = 0; i < cpd.m_pc_verts.size(); i ++) {
        Point_2 p = cpd.m_pc_verts[i];
        V_power_crust(i, 0) = CGAL::to_double(p.x());
        V_power_crust(i, 1) = CGAL::to_double(p.y());
    }
    for (size_t i = 0; i < cpd.m_pd_out_edges.size(); i ++) {
        E_power_diagram_out(i, 0) = cpd.m_pd_out_edges[i].first;
        E_power_diagram_out(i, 1) = cpd.m_pd_out_edges[i].second;
    }
    for (size_t i = 0; i < cpd.m_pd_in_edges.size(); i ++) {
        E_power_diagram_in(i, 0) = cpd.m_pd_in_edges[i].first;
        E_power_diagram_in(i, 1) = cpd.m_pd_in_edges[i].second;
    }
    for (size_t i = 0; i < cpd.m_pc_edges.size(); i ++) {
        E_power_crust(i, 0) = cpd.m_pc_edges[i].first;
        E_power_crust(i, 1) = cpd.m_pc_edges[i].second;
    }
}
