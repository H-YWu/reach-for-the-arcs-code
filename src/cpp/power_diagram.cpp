#include "power_diagram.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>

#include <unordered_map>
#include <iterator>
#include <stdexcept>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_2<K> Regular_triangulation_2;
typedef K::Point_2 Point_2;
typedef K::Weighted_point_2 Weighted_point_2;
typedef K::Iso_rectangle_2 Iso_rectangle_2;
typedef K::Segment_2 Segment_2;
typedef K::Ray_2 Ray_2;
typedef K::Line_2 Line_2;

struct Cropped_power_from_regular2 {
    std::vector<std::pair<Point_2, Point_2>> m_cropped_pd_edges;
    std::vector<bool> m_is_edge_separator;

    Iso_rectangle_2 m_bbox;
    Regular_triangulation_2 m_rt2;
    std::unordered_map<Point_2, int> m_psigns;

    Cropped_power_from_regular2(
        const Iso_rectangle_2& bbox,
        const Regular_triangulation_2& rt2,
        const std::unordered_map<Point_2, int>& psigns
    ) : m_bbox(bbox), m_rt2(rt2), m_psigns(psigns) {
        crop_and_extract_all_segments();
    }

    void crop_and_extract_all_segments() {
        for (const auto& e : m_rt2.finite_edges()) {
            CGAL::Object de = m_rt2.dual(e);
            bool is_seg = false;
            if (const Segment_2* seg = CGAL::object_cast<Segment_2>(&de)) {
                is_seg = crop_and_extract_segment(*seg);
            } else if (const Ray_2* ray = CGAL::object_cast<Ray_2>(&de)) {
                is_seg = crop_and_extract_segment(*ray);
            } else if (const Line_2* line = CGAL::object_cast<Line_2>(&de)) {
                is_seg = crop_and_extract_segment(*line);
            }
            if (is_seg) {
                auto vh1 = e.first->vertex((e.second + 1) % 3);
                auto vh2 = e.first->vertex((e.second + 2) % 3);
                Point_2 p1 = vh1->point().point();
                Point_2 p2 = vh2->point().point();
                auto get_sign = [&](const Point_2& p) -> size_t {
                    auto it = m_psigns.find(p);
                    if (it == m_psigns.end()) {
                        throw std::runtime_error("No sign assigned to this point!");
                    }
                    return it->second;
                };
                int s1 = get_sign(p1), s2 = get_sign(p2);
                if (s1 * s2 == -1) {
                    m_is_edge_separator.emplace_back(true);
                }
                else {
                    m_is_edge_separator.emplace_back(false);
                }
            } 
        }
    }

    template <class RSL>
    bool crop_and_extract_segment(const RSL& rsl) {
        CGAL::Object obj = CGAL::intersection(rsl, m_bbox);
        const Segment_2* s = CGAL::object_cast<Segment_2>(&obj);
        if (s) {
            m_cropped_pd_edges.emplace_back(s->source(), s->target());
            return true;
        }
        return false;
    }
};

void power_diagram_2d(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sdf_data,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & E,
    BoolVector & Es
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
    Cropped_power_from_regular2 pd(bbox, rt2, psigns);

    // Build V and E
    std::unordered_map<Point_2, size_t> index_of_point;
    std::vector<std::array<double, 2>> vec_V;
    std::vector<std::array<size_t, 2>> vec_E;
    for (const auto& segment : pd.m_cropped_pd_edges) {
        auto insert_point = [&](const Point_2& p) -> size_t {
            auto it = index_of_point.find(p);
            if (it == index_of_point.end()) {
                size_t index = vec_V.size();
                vec_V.push_back({p.x(), p.y()});
                index_of_point[p] = index;
                return index;
            }
            return it->second;
        };

        size_t index1 = insert_point(segment.first);
        size_t index2 = insert_point(segment.second);
        vec_E.push_back({index1, index2});
    }
    // Convert std::vector to Eigen::Matrix
    V.resize(vec_V.size(), 2);
    E.resize(vec_E.size(), 2);
    Es.resize(pd.m_cropped_pd_edges.size(), 1);
    for (size_t i = 0; i < vec_V.size(); i ++) {
        V(i, 0) = vec_V[i][0];
        V(i, 1) = vec_V[i][1];
    }
    for (size_t i = 0; i < vec_E.size(); i ++) {
        E(i, 0) = vec_E[i][0];
        E(i, 1) = vec_E[i][1];
    }
    for (size_t i = 0; i < pd.m_is_edge_separator.size(); i ++) {
        Es(i) = pd.m_is_edge_separator[i];
        Es(i) = pd.m_is_edge_separator[i];
    }
}
