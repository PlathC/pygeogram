#include <memory>

//
#include "disable_warnings.hpp"
//
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/delaunay/delaunay_2d.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/voronoi/RVD.h>
//
#include "enable_warnings.hpp"
//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

using DoubleArray  = nb::ndarray<double, nb::numpy, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using VectorArray  = nb::ndarray<double, nb::numpy, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>;
using SimplexArray = nb::ndarray<GEO::index_t, nb::numpy, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>;

struct Voronoi
{
    Voronoi(const VectorArray& iSeeds, const DoubleArray& iWeights, const VectorArray& domainVertices,
            const SimplexArray& domainSimplices)
        : dimension(static_cast<uint32_t>(iSeeds.shape(1))), //
          seedNb(static_cast<uint32_t>(iSeeds.shape(0))),    //
          isWeighted(iWeights.is_valid())                    //
    {
        GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);
        GEO::CmdLine::set_arg("dbg:delaunay", true);
        GEO::CmdLine::set_arg("dbg:delaunay_verbose", true);
        GEO::CmdLine::set_arg("dbg:delaunay_benchmark", false);

        GEO::CmdLine::import_arg_group("standard");
        GEO::CmdLine::import_arg_group("algo");

        geo_assert(iSeeds.shape(1) == 2 || iSeeds.shape(1) == 3);
        geo_assert(iSeeds.shape(0) > 0);
        geo_assert(!isWeighted || iSeeds.shape(0) == iWeights.shape(0));
        geo_assert(domainVertices.shape(1) == dimension);
        geo_assert(domainSimplices.shape(1) == dimension + 1);

        // Copy buffers
        seeds.resize(seedNb * dimension);
        std::memcpy(seeds.data(), iSeeds.data(), seedNb * dimension * sizeof(double));

        printf("is weighted -> %d\n", isWeighted);
        if (isWeighted)
        {
            weights.resize(seedNb);
            std::memcpy(weights.data(), iWeights.data(), seedNb * sizeof(double));
        }

        // Compute triangulation
        std::string name = "default";
        if (dimension == 2)
            name = isWeighted ? "BPOW2d" : "BDEL2d";
        else if (dimension == 3)
            name = isWeighted ? "BPOW" : "PDEL";

        GEO::Delaunay_var impl  = GEO::Delaunay::create(GEO::coord_index_t(dimension + (isWeighted ? 1 : 0)), name);
        auto              input = std::vector((dimension + (isWeighted ? 1 : 0)) * seedNb, 0.);
        if (isWeighted)
        {
            double maxWeight = -std::numeric_limits<double>::max();
            for (uint32_t i = 0; i < seedNb; ++i)
                maxWeight = std::max(maxWeight, weights[i]);

            const uint32_t stride = dimension + 1;
            for (uint32_t i = 0; i < seedNb; ++i)
            {
                for (uint32_t d = 0; d < dimension; ++d)
                    input[stride * i + d] = seeds[stride * i + d];

                input[stride * i + dimension] = std::sqrt(maxWeight - weights[i]);
            }
        }
        else
        {
            std::memcpy(input.data(), seeds.data(), seedNb * dimension * sizeof(double));
            for (uint32_t i = 0; i < seedNb; ++i)
            {
                for (uint32_t d = 0; d < dimension; ++d)
                    printf("%.6f, ", input[i * dimension + d]);

                printf("\n");
            }
        }

        GEO::Mesh      domain;
        const uint32_t domainVertexNb  = static_cast<GEO::index_t>(domainVertices.shape(0) / dimension);
        const uint32_t domainSimplexNb = static_cast<uint32_t>(domainSimplices.shape(0));

        GEO::vector<GEO::index_t> domainSimplicesCopy = GEO::vector<GEO::index_t>(domainSimplexNb * dimension);
        for (uint32_t i = 0; i < domainSimplexNb; i++)
        {
            for (uint32_t d = 0; d < dimension; ++d)
                domainSimplicesCopy[i * dimension + d] = domainSimplices(i, d);
        }

        GEO::vector<double> domainVerticesCopy(domainVertexNb * dimension);
        for (uint32_t v = 0; v < domainVertexNb; ++v)
        {
            for (uint32_t d = 0; d < dimension; ++d)
                domainVerticesCopy[v * dimension + d] = domainVertices(v, d);
        }

        if (impl->dimension() == 3)
            domain.cells.assign_tet_mesh(dimension, domainVerticesCopy, domainSimplicesCopy, true);
        else if (impl->dimension() == 2)
            domain.facets.assign_triangle_mesh(dimension, domainVerticesCopy, domainSimplicesCopy, true);
        else
            geo_assert_not_reached;

        // Compute voronoi diagram
        GEO::RestrictedVoronoiDiagram_var RVD = GEO::RestrictedVoronoiDiagram::create(impl, &domain);
        RVD->set_volumetric(true);

        GEO::Mesh resultMesh;
        impl->set_vertices(GEO::index_t(seedNb), input.data());
        RVD->compute_RVD(resultMesh);

        // Output voronoi diagram
        printf("impl->dimension() -> %d\n", impl->dimension());
        printf("impl->nb_cells() -> %d\n", impl->nb_cells());
        printf("Found %d vertices of dimension %d\n", resultMesh.vertices.nb(), resultMesh.vertices.dimension());

        vertices.resize(resultMesh.vertices.nb() * dimension);
        for (uint32_t v = 0; v < resultMesh.vertices.nb(); v++)
        {
            for (uint32_t d = 0; d < dimension; ++d)
                vertices[v * dimension + d] = resultMesh.vertices.point_ptr(v)[d];
        }

        // One or the other of the following structure should be kept depending on the dimension
        triangle_vertices.resize(resultMesh.facets.nb() * 3);
        std::memcpy(triangle_vertices.data(), resultMesh.facet_corners.vertex_index_ptr(0),
                    sizeof(GEO::index_t) * triangle_vertices.size());

        // adjacency seems to be only available for triangles
        triangle_adjacency.resize(resultMesh.facets.nb() * 3);
        std::memcpy(triangle_adjacency.data(), resultMesh.facet_corners.adjacent_facet_ptr(0),
                    sizeof(GEO::index_t) * triangle_adjacency.size());

        // GEO::Attribute<GEO::index_t> facet_region_attr(resultMesh.facets.attributes(), "region");
        // for (GEO::index_t f = 0; f < resultMesh.facets.nb(); ++f)
        //     triangle_regions[f] = facet_region_attr[f];

        // - tetrahedra
        tet_vertices.resize(resultMesh.cells.nb() * 4);
        std::memcpy(tet_vertices.data(), resultMesh.cell_corners.vertex_index_ptr(0),
                    sizeof(GEO::index_t) * tet_vertices.size());

        // GEO::Attribute<GEO::index_t> cell_region_attr(resultMesh.cells.attributes(), "region");
        // for (GEO::index_t c = 0; c < resultMesh.cells.nb(); ++c)
        //     tet_regions[c] = cell_region_attr[c];
    }

    // Settings
    uint32_t dimension;
    uint32_t seedNb;
    bool     isWeighted;

    // Input data
    std::vector<double> seeds;
    std::vector<double> weights;

    // Diagram data
    std::vector<double> vertices;

    std::vector<GEO::index_t> triangle_vertices;
    std::vector<GEO::index_t> triangle_regions;
    std::vector<GEO::index_t> triangle_adjacency;

    std::vector<GEO::index_t> tet_vertices;
    std::vector<GEO::index_t> tet_regions;
};

NB_MODULE(geogram_ext, m)
{
    nb::class_<Voronoi>(m, "Voronoi")
        .def(nb::init<const VectorArray&, const DoubleArray&, const VectorArray&, const SimplexArray&>(), "seeds"_a,
             "weights"_a = nb::none(), "domain_vertices"_a, "domain_simplices"_a)
        .def_ro("dimension", &Voronoi::dimension)
        .def_prop_ro("seeds",
                     [](Voronoi& diagram) {
                         return VectorArray(diagram.seeds.data(),
                                            {diagram.seeds.size() / diagram.dimension, diagram.dimension});
                     })
        .def_ro("weights", &Voronoi::weights)
        .def_prop_ro("q",
                     [](Voronoi& diagram) {
                         return VectorArray(diagram.vertices.data(),
                                            {diagram.vertices.size() / diagram.dimension, diagram.dimension});
                     })
        .def_prop_ro("t",
                     [](Voronoi& diagram) {
                         return SimplexArray(diagram.triangle_vertices.data(),
                                             {diagram.triangle_vertices.size() / 3, 3});
                     })
        .def_ro("tseed", &Voronoi::triangle_regions)
        .def_prop_ro("tadj", [](Voronoi& diagram) {
            return SimplexArray(diagram.triangle_adjacency.data(), {diagram.triangle_adjacency.size() / 3, 3});
        });

    m.def_submodule("domain") //
        .def("quad",
             []() {
                 std::vector vertices = {
                     0., 0., //
                     0., 1., //
                     1., 0., //
                     1., 1., //
                 };
                 std::vector triangles = {
                     0, 1, 2, 2, 1, 3,
                 };

                 return nb::make_tuple(VectorArray(vertices.data(), {vertices.size() / 2, 2}).cast(),
                                       SimplexArray(triangles.data(), {triangles.size() / 3, 3}).cast());
             })
        .def("load", [](const std::string& path, const uint32_t dimension) {
            geo_assert(dimension == 2 || dimension == 3);

            GEO::Mesh mesh;
            GEO::mesh_load(path, mesh);

            // Vertices
            std::vector<double> vertices = std::vector(mesh.vertices.nb() * mesh.vertices.dimension(), 0.);
            std::memcpy(vertices.data(), mesh.vertices.point_ptr(0), vertices.size() * sizeof(double));

            // Domain simplices
            std::vector<GEO::index_t> simplices = {};
            if (dimension == 2)
            {
                mesh.facets.triangulate(); // required?
                simplices.resize(mesh.facets.nb() * 3);
                std::memcpy(simplices.data(), mesh.facet_corners.vertex_index_ptr(0),
                            simplices.size() * sizeof(GEO::index_t));
            }
            else if (dimension == 3)
            {
                simplices.resize(mesh.cells.nb() * 4);
                std::memcpy(simplices.data(), mesh.cell_corners.vertex_index_ptr(0),
                            simplices.size() * sizeof(GEO::index_t));
            }
            else
            {
                geo_assert_not_reached;
            }

            return nb::make_tuple(
                VectorArray(vertices.data(), {vertices.size() / dimension, dimension}).cast(),
                SimplexArray(simplices.data(), {simplices.size() / (dimension + 1), dimension + 1}).cast());
        });
}
