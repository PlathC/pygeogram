#include <memory>

//
#include "disable_warnings.hpp"
//
#include <geogram/basic/command_line.h>
#include <geogram/delaunay/delaunay_2d.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/mesh/mesh_io.h>

//
#include "enable_warnings.hpp"
//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/vector.h>

#include "geogram/voronoi/RVD.h"

namespace nb = nanobind;
using namespace nb::literals;
using Points2d = nb::ndarray<double, nb::numpy, nb::shape<-1, 2>, nb::device::cpu, nb::c_contig>;
using Points3d = nb::ndarray<double, nb::numpy, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using Weights  = nb::ndarray<double, nb::numpy, nb::shape<-1>, nb::device::cpu, nb::c_contig>;

struct PowerDiagram2d
{
    PowerDiagram2d(const Points2d& centers, const Weights& wweights)
    {
        GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);
        GEO::CmdLine::set_arg("dbg:delaunay", false);
        GEO::CmdLine::set_arg("dbg:delaunay_verbose", false);
        GEO::CmdLine::set_arg("dbg:delaunay_benchmark", false);

        assert(centers.shape(0) == weights.shape(0));
        const std::size_t siteNb = centers.shape(0);

        impl = GEO::Delaunay::create(GEO::coord_index_t(3), "BPOW2d");
        impl->set_keeps_infinite(true);
        impl->set_stores_cicl(true);

        double maxWeight = -std::numeric_limits<double>::max();
        for (uint32_t i = 0; i < siteNb; ++i)
            maxWeight = std::max(maxWeight, wweights(i));

        points.resize(siteNb * 2);
        weights.resize(siteNb);

        std::vector<double> pointsAndWeights = std::vector(3 * siteNb, 0.);
        for (uint32_t i = 0; i < siteNb; ++i)
        {
            pointsAndWeights[3 * i + 0] = centers(i, 0);
            pointsAndWeights[3 * i + 1] = centers(i, 1);
            pointsAndWeights[3 * i + 2] = std::sqrt(maxWeight - wweights(i));

            points[2 * i + 0] = centers(i, 0);
            points[2 * i + 1] = centers(i, 1);
            weights[i]        = wweights(i);
        }

        impl->set_vertices(GEO::index_t(siteNb), pointsAndWeights.data());

        const uint32_t tetNb = impl->nb_cells();
        triangles.resize(tetNb * 3);

        for (GEO::index_t t = 0; t < tetNb; ++t)
        {
            triangles[t * 3 + 0] = GEO::index_t(impl->cell_vertex(t, 0));
            triangles[t * 3 + 1] = GEO::index_t(impl->cell_vertex(t, 1));
            triangles[t * 3 + 2] = GEO::index_t(impl->cell_vertex(t, 2));
        }
    }

    std::array<double, 3> bisector(uint32_t i, uint32_t j)
    {
        const GEO::vec2 pi = GEO::vec2(&points[i * 3]);
        const GEO::vec2 pj = GEO::vec2(&points[j * 3]);

        const double wij = -.5 * (dot(pj, pj) - dot(pi, pi) - weights[j] + weights[i]);
        return {pj.x - pi.x, pj.y - pi.y, wij};
    }

    std::array<double, 2> vertex(uint32_t v)
    {
        const uint32_t i = triangles[v * 4 + 0];
        const uint32_t j = triangles[v * 4 + 1];
        const uint32_t k = triangles[v * 4 + 2];

        const auto pij = bisector(i, j);
        const auto pik = bisector(i, k);

        const GEO::mat2 m  = {{
            pij[0], pij[1], //
            pik[0], pik[1], //
        }};
        const GEO::mat2 im = m.inverse();

        const GEO::vec2 vertex = im * GEO::vec2(-pij[2], -pik[2]);
        return {vertex.x, vertex.y};
    }

    std::vector<double> points;
    std::vector<double> weights;

    GEO::Delaunay_var     impl;
    std::vector<uint32_t> triangles;
};

struct PowerDiagram3d
{
    PowerDiagram3d(const Points3d& centers, const Weights& wweights)
    {
        GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);
        GEO::CmdLine::set_arg("dbg:delaunay", false);
        GEO::CmdLine::set_arg("dbg:delaunay_verbose", false);
        GEO::CmdLine::set_arg("dbg:delaunay_benchmark", false);

        assert(centers.shape(0) == wweights.shape(0));
        const std::size_t siteNb = centers.shape(0);

        impl = GEO::Delaunay::create(GEO::coord_index_t(4), "BPOW");
        impl->set_keeps_infinite(true);

        double maxWeight = -std::numeric_limits<double>::max();
        for (uint32_t i = 0; i < siteNb; ++i)
            maxWeight = std::max(maxWeight, wweights(i));

        points.resize(siteNb * 3);
        weights.resize(siteNb);

        std::vector<double> pointsAndWeights = std::vector(4 * siteNb, 0.);
        for (uint32_t i = 0; i < siteNb; ++i)
        {
            pointsAndWeights[4 * i + 0] = centers(i, 0);
            pointsAndWeights[4 * i + 1] = centers(i, 1);
            pointsAndWeights[4 * i + 2] = centers(i, 2);
            pointsAndWeights[4 * i + 3] = std::sqrt(maxWeight - wweights(i));

            points[3 * i + 0] = centers(i, 0);
            points[3 * i + 1] = centers(i, 1);
            points[3 * i + 2] = centers(i, 2);
            weights[i]        = wweights(i);
        }

        impl->set_vertices(GEO::index_t(siteNb), pointsAndWeights.data());

        const uint32_t tetNb = impl->nb_cells();
        tetrahedra.resize(tetNb * 4);

        for (GEO::index_t t = 0; t < tetNb; ++t)
        {
            tetrahedra[t * 4 + 0] = GEO::index_t(impl->cell_vertex(t, 0));
            tetrahedra[t * 4 + 1] = GEO::index_t(impl->cell_vertex(t, 1));
            tetrahedra[t * 4 + 2] = GEO::index_t(impl->cell_vertex(t, 2));
            tetrahedra[t * 4 + 3] = GEO::index_t(impl->cell_vertex(t, 3));
        }
    }

    std::array<double, 4> bisector(uint32_t i, uint32_t j)
    {
        const GEO::vec3 pi = GEO::vec3(&points[i * 3]);
        const GEO::vec3 pj = GEO::vec3(&points[j * 3]);

        const double wij = -.5 * (dot(pj, pj) - dot(pi, pi) - weights[j] + weights[i]);
        return {pj.x - pi.x, pj.y - pi.y, pj.z - pi.z, wij};
    }

    std::array<double, 3> vertex(uint32_t v)
    {
        const uint32_t i = tetrahedra[v * 4 + 0];
        const uint32_t j = tetrahedra[v * 4 + 1];
        const uint32_t k = tetrahedra[v * 4 + 2];
        const uint32_t l = tetrahedra[v * 4 + 3];

        const auto pij = bisector(i, j);
        const auto pik = bisector(i, k);
        const auto pil = bisector(i, l);

        const GEO::mat3 m  = {{
            pij[0], pij[1], pij[2], //
            pik[0], pik[1], pik[2], //
            pil[0], pil[1], pil[2], //
        }};
        const GEO::mat3 im = m.inverse();

        const GEO::vec3 vertex = im * GEO::vec3(-pij[3], -pik[3], -pil[3]);
        return {vertex.x, vertex.y, vertex.z};
    }

    std::vector<double> points;
    std::vector<double> weights;

    GEO::Delaunay_var     impl;
    std::vector<uint32_t> tetrahedra;
};

NB_MODULE(geogram_ext, m)
{
    nb::class_<PowerDiagram2d>(m, "PowerDiagram2d")
        .def(nb::init<const Points2d&, const Weights&>(), "centers"_a, "weights"_a)
        .def("triangles",
             [](PowerDiagram2d& diagram) {
                 return nb::ndarray<nb::numpy, uint32_t>( //
                     diagram.triangles.data(), {diagram.triangles.size() / 3, 3}, nb::handle());
             })
        .def("size", [](const PowerDiagram2d& diagram) { return diagram.triangles.size() / 3; })
        .def("vertex", &PowerDiagram2d::vertex);

    nb::class_<PowerDiagram3d>(m, "PowerDiagram3d")
        .def(nb::init<const Points3d&, const Weights&>(), "centers"_a, "weights"_a)
        .def("tetrahedra",
             [](PowerDiagram3d& diagram) {
                 return nb::ndarray<nb::numpy, uint32_t>( //
                     diagram.tetrahedra.data(), {diagram.tetrahedra.size() / 4, 4});
             })
        .def("size", [](const PowerDiagram3d& diagram) { return diagram.tetrahedra.size() / 4; })
        .def("vertex", &PowerDiagram3d::vertex);
}
