#include <memory>

//
#include "disable_warnings.hpp"
//
#include <geogram/basic/command_line.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/mesh/mesh_io.h>
//
#include "enable_warnings.hpp"
//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using WeightedSite3dList = nb::ndarray<double, nb::numpy, nb::shape<-1, 4>, nb::device::cpu, nb::c_contig>;

struct PowerDiagram3d
{
    PowerDiagram3d(const WeightedSite3dList& sites, bool abortIfEmpty = false)
    {
        GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);
        GEO::CmdLine::set_arg("dbg:delaunay", false);
        GEO::CmdLine::set_arg("dbg:delaunay_verbose", false);
        GEO::CmdLine::set_arg("dbg:delaunay_benchmark", false);

        const std::size_t siteNb = sites.shape(0);

        std::vector<double> points  = std::vector(siteNb * 3, 0.);
        std::vector<double> weights = std::vector(siteNb, 0.);
        for (std::size_t i = 0; i < siteNb; i++)
        {
            points[i * 3 + 0] = sites(i, 0);
            points[i * 3 + 1] = sites(i, 1);
            points[i * 3 + 2] = sites(i, 2);
            weights[i]        = sites(i, 3);
        }

        const auto impl = std::unique_ptr<GEO::PeriodicDelaunay3d>(new GEO::PeriodicDelaunay3d(false, 10.));
        impl->use_exact_predicates_for_convex_cell(true);
        impl->set_keeps_infinite(true);

        impl->set_vertices(GEO::index_t(siteNb), points.data());
        impl->set_weights(weights.data());
        impl->compute();

        hasEmptyCells = impl->has_empty_cells();
        if (impl->has_empty_cells() && abortIfEmpty)
            return;

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

    std::vector<uint32_t> tetrahedra;
    bool                  hasEmptyCells = false;
};

NB_MODULE(grampy_ext, m)
{
    nb::class_<PowerDiagram3d>(m, "PowerDiagram3d")
        .def(nb::init<const WeightedSite3dList&, bool>(), "sites"_a, "abord_if_empty"_a = false)
        .def("tetrahedra",
             [](PowerDiagram3d& diagram) {
                 return nb::ndarray<nb::numpy, uint32_t>( //
                     diagram.tetrahedra.data(), {diagram.tetrahedra.size() / 4, 4}, nb::handle());
             })
        .def_ro("has_empty", &PowerDiagram3d::hasEmptyCells)
        .def("size", [](PowerDiagram3d& diagram) { return diagram.tetrahedra.size() / 4; });
}
