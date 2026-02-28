#include <memory>
#include <span>

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

#include "geogram/mesh/mesh_tetrahedralize.h"
#include "geogram/third_party/PoissonRecon/MarchingCubes.h"

namespace nb = nanobind;
using namespace nb::literals;

using DoubleArray  = nb::ndarray<double, nb::numpy, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using VectorArray  = nb::ndarray<double, nb::numpy, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>;
using SimplexArray = nb::ndarray<GEO::index_t, nb::numpy, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>;

void surfaceToArray(GEO::Mesh& mesh, std::vector<double>& oVertices, std::vector<GEO::index_t>& oSimplices)
{
    const uint32_t dimension = mesh.vertices.dimension();

    mesh.facets.triangulate(); // required?
    oVertices.resize(mesh.vertices.nb() * dimension);
    std::memcpy(oVertices.data(), mesh.vertices.point_ptr(0), oVertices.size() * sizeof(double));

    oSimplices.resize(mesh.facets.nb() * 3);
    std::memcpy(oSimplices.data(), mesh.facet_corners.vertex_index_ptr(0), oSimplices.size() * sizeof(GEO::index_t));
}

void volumeToArray(GEO::Mesh& mesh, std::vector<double>& oVertices, std::vector<GEO::index_t>& oSimplices)
{
    geo_assert((mesh.vertices.dimension()) == 3);

    oVertices.resize(mesh.vertices.nb() * 3);
    std::memcpy(oVertices.data(), mesh.vertices.point_ptr(0), oVertices.size() * sizeof(double));

    oSimplices.resize(mesh.cells.nb() * 4);
    std::memcpy(oSimplices.data(), mesh.cell_corners.vertex_index_ptr(0), oSimplices.size() * sizeof(GEO::index_t));
}

void arrayToSurface(const std::span<const double> iVertices, const std::span<const GEO::index_t> iSimplices,
                    GEO::Mesh& mesh, uint32_t dimension = 3)
{
    geo_assert((iVertices.size() % dimension) == 0);

    const uint32_t vertexNb  = static_cast<GEO::index_t>(iVertices.size() / dimension);
    const uint32_t simplexNb = static_cast<uint32_t>(iSimplices.size() / 3);

    GEO::vector<GEO::index_t> oSimplicies = GEO::vector<GEO::index_t>(simplexNb * 3);
    std::memcpy(oSimplicies.data(), iSimplices.data(), oSimplicies.size() * sizeof(GEO::index_t));

    GEO::vector<double> oVertices(vertexNb * dimension);
    std::memcpy(oVertices.data(), iVertices.data(), oVertices.size() * sizeof(double));

    mesh.facets.assign_triangle_mesh(dimension, oVertices, oSimplicies, true);
}

void arrayToVolume(const std::span<const double> iVertices, const std::span<const GEO::index_t> iSimplices,
                   GEO::Mesh& mesh)
{
    geo_assert(iVertices.size() % 3 == 0);
    const uint32_t vertexNb  = static_cast<GEO::index_t>(iVertices.size() / 3);
    const uint32_t simplexNb = static_cast<uint32_t>(iSimplices.size() / 4);

    GEO::vector<double> oVertices(vertexNb * 3);
    std::memcpy(oVertices.data(), iVertices.data(), oVertices.size() * sizeof(double));

    GEO::vector<GEO::index_t> oSimplices = GEO::vector<GEO::index_t>(simplexNb * 4);
    std::memcpy(oSimplices.data(), iSimplices.data(), oSimplices.size() * sizeof(GEO::index_t));

    mesh.cells.assign_tet_mesh(3, oVertices, oSimplices, true);
}

struct Voronoi
{
    Voronoi(const VectorArray& iSeeds, const DoubleArray& iWeights, const VectorArray& domainVertices,
            const SimplexArray& domainSimplices)
        : dimension(static_cast<uint32_t>(iSeeds.shape(1))), //
          seedNb(static_cast<uint32_t>(iSeeds.shape(0))),    //
          isWeighted(iWeights.is_valid())                    //
    {
        geo_assert(iSeeds.shape(1) == 2 || iSeeds.shape(1) == 3);
        geo_assert(iSeeds.shape(0) > 0);
        geo_assert(!isWeighted || iSeeds.shape(0) == iWeights.shape(0));
        geo_assert(domainVertices.shape(1) == dimension);
        geo_assert(domainSimplices.shape(1) == dimension + 1);

        // Copy buffers
        seeds.resize(seedNb * dimension);
        std::memcpy(seeds.data(), iSeeds.data(), seedNb * dimension * sizeof(double));

        if (isWeighted)
        {
            // Weighted Delaunay is not supported yet.
            weights.resize(seedNb);
            std::memcpy(weights.data(), iWeights.data(), seedNb * sizeof(double));
        }

        // Delaunay needs a copy
        auto input = std::vector((dimension + (isWeighted ? 1 : 0)) * seedNb, 0.);
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
        }

        if (dimension == 2)
        {
            std::string name = isWeighted ? "BPOW2d" : "BDEL2d";
            if (isWeighted)
                geo_assert(false);

            GEO::Mesh domain;
            arrayToSurface({domainVertices.data(), domainVertices.size()},
                           {domainSimplices.data(), domainSimplices.size()}, domain, 2);

            GEO::Delaunay_var                 delaunay = GEO::Delaunay::create(dimension + (isWeighted ? 1 : 0));
            GEO::RestrictedVoronoiDiagram_var RVD      = GEO::RestrictedVoronoiDiagram::create(delaunay, &domain);
            delaunay->set_vertices(seedNb, input.data());

            GEO::Mesh resultMesh;
            RVD->compute_RVD(resultMesh, 0, false, false);

            geo_assert(resultMesh.vertices.dimension() == 2);
            vertices.resize(resultMesh.vertices.nb() * 2);
            std::memcpy(vertices.data(), resultMesh.vertices.point_ptr(0), sizeof(double) * vertices.size());

            simplex_vertices.resize(resultMesh.facets.nb() * 3);
            std::memcpy(simplex_vertices.data(), resultMesh.facet_corners.vertex_index_ptr(0),
                        sizeof(GEO::index_t) * simplex_vertices.size());

            simplex_adjacency.resize(resultMesh.facets.nb() * 3);
            std::memcpy(simplex_adjacency.data(), resultMesh.facet_corners.adjacent_facet_ptr(0),
                        sizeof(GEO::index_t) * simplex_adjacency.size());

            vertices.resize(resultMesh.vertices.nb() * 2);
            std::memcpy(vertices.data(), resultMesh.vertices.point_ptr(0), sizeof(double) * vertices.size());

            simplex_regions.resize(resultMesh.facets.nb());
            GEO::Attribute<GEO::index_t> facet_region_attr(resultMesh.facets.attributes(), "region");
            for (GEO::index_t f = 0; f < resultMesh.facets.nb(); ++f)
                simplex_regions[f] = facet_region_attr[f];

            return;
        }
        else if (dimension == 3)
        {
            // Compute triangulation
            std::string name = isWeighted ? "BPOW" : "PDEL";

            GEO::Mesh domain;
            arrayToVolume({domainVertices.data(), domainVertices.size()},
                          {domainSimplices.data(), domainSimplices.size()}, domain);

            if (isWeighted)
                domain.vertices.set_dimension(4);

            GEO::Delaunay_var                 delaunay = GEO::Delaunay::create(dimension + (isWeighted ? 1 : 0));
            GEO::RestrictedVoronoiDiagram_var RVD      = GEO::RestrictedVoronoiDiagram::create(delaunay, &domain);
            delaunay->set_vertices(seedNb, input.data());

            RVD->set_volumetric(true);

            GEO::Mesh resultMesh;
            RVD->compute_RVD(resultMesh, 0, false, false);
            if (isWeighted)
            {
                // domain.vertices.set_dimension(3);
                resultMesh.vertices.set_dimension(3);
            }

            geo_assert(resultMesh.vertices.dimension() == 3);
            vertices.resize(resultMesh.vertices.nb() * 3);
            for (uint32_t v = 0; v < resultMesh.vertices.nb(); ++v)
                std::memcpy(vertices.data() + v * 3, resultMesh.vertices.point_ptr(v), sizeof(double) * 3);

            simplex_vertices.resize(resultMesh.cells.nb() * 4);
            std::memcpy(simplex_vertices.data(), resultMesh.cell_corners.vertex_index_ptr(0),
                        sizeof(GEO::index_t) * simplex_vertices.size());

            // How to compute adjacency? -> Copy what is done for triangles?
            // simplex_adjacency.resize(resultMesh.facets.nb() * 4);
            // std::memcpy(simplex_adjacency.data(), resultMesh.facet_corners.adjacent_facet_ptr(0),
            //             sizeof(GEO::index_t) * simplex_adjacency.size());

            simplex_regions.resize(resultMesh.cells.nb());
            GEO::Attribute<GEO::index_t> facet_region_attr(resultMesh.cells.attributes(), "region");
            for (GEO::index_t f = 0; f < resultMesh.cells.nb(); ++f)
                simplex_regions[f] = facet_region_attr[f];
        }
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

    std::vector<GEO::index_t> simplex_vertices;
    std::vector<GEO::index_t> simplex_regions;
    std::vector<GEO::index_t> simplex_adjacency;
};

NB_MODULE(geogram_ext, m)
{
    m.def("initialize", []() {
        GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);
        GEO::CmdLine::set_arg("dbg:delaunay", false);
        GEO::CmdLine::set_arg("dbg:delaunay_verbose", false);
        GEO::CmdLine::set_arg("dbg:delaunay_benchmark", false);

        GEO::CmdLine::import_arg_group("standard");
        GEO::CmdLine::import_arg_group("algo");
        GEO::CmdLine::import_arg_group("remesh");

        GEO::Logger::instance()->unregister_all_clients();
    });

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
                         return SimplexArray(
                             diagram.simplex_vertices.data(),
                             {diagram.simplex_vertices.size() / (diagram.dimension + 1), (diagram.dimension + 1)});
                     })
        .def_prop_ro("tseed",
                     [](Voronoi& diagram) {
                         return SimplexArray(diagram.simplex_regions.data(), {diagram.simplex_regions.size()});
                     })
        .def_prop_ro("tadj", [](Voronoi& diagram) {
            return SimplexArray(diagram.simplex_adjacency.data(),
                                {diagram.simplex_adjacency.size() / (diagram.dimension + 1), (diagram.dimension + 1)});
        });

    m.def_submodule("shape") //
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
        .def("cube", []() {
            GEO::Mesh mesh;

            const double x1 = 0., y1 = 0., z1 = 0.;
            const double x2 = 1., y2 = 1., z2 = 1.;

            GEO::index_t v0 = mesh.vertices.create_vertex(GEO::vec3(x1, y1, z1).data());
            GEO::index_t v1 = mesh.vertices.create_vertex(GEO::vec3(x1, y1, z2).data());
            GEO::index_t v2 = mesh.vertices.create_vertex(GEO::vec3(x1, y2, z1).data());
            GEO::index_t v3 = mesh.vertices.create_vertex(GEO::vec3(x1, y2, z2).data());
            GEO::index_t v4 = mesh.vertices.create_vertex(GEO::vec3(x2, y1, z1).data());
            GEO::index_t v5 = mesh.vertices.create_vertex(GEO::vec3(x2, y1, z2).data());
            GEO::index_t v6 = mesh.vertices.create_vertex(GEO::vec3(x2, y2, z1).data());
            GEO::index_t v7 = mesh.vertices.create_vertex(GEO::vec3(x2, y2, z2).data());

            mesh.facets.create_quad(v3, v7, v6, v2);
            mesh.facets.create_quad(v0, v1, v3, v2);
            mesh.facets.create_quad(v1, v5, v7, v3);
            mesh.facets.create_quad(v5, v4, v6, v7);
            mesh.facets.create_quad(v0, v4, v5, v1);
            mesh.facets.create_quad(v2, v6, v4, v0);
            mesh.facets.connect();
            mesh.facets.triangulate();

            // Conversion
            std::vector<double>       vertices  = {};
            std::vector<GEO::index_t> triangles = {};
            surfaceToArray(mesh, vertices, triangles);

            return nb::make_tuple(VectorArray(vertices.data(), {vertices.size() / 3, 3}).cast(),
                                  SimplexArray(triangles.data(), {triangles.size() / 3, 3}).cast());
        });

    m.def_submodule("mesh") //
        .def("tetrahedralize",
             [](const VectorArray vertices, const SimplexArray triangles) {
                 geo_assert(vertices.shape(1) == 3);
                 geo_assert(triangles.shape(1) == 3);

                 GEO::Mesh mesh;
                 arrayToSurface({vertices.data(), vertices.size()}, {triangles.data(), triangles.size()}, mesh);

                 GEO::mesh_tetrahedralize(mesh);

                 // Output
                 std::vector<double>       oVertices  = {};
                 std::vector<GEO::index_t> oSimplices = {};
                 volumeToArray(mesh, oVertices, oSimplices);

                 return nb::make_tuple(VectorArray(oVertices.data(), {oVertices.size() / 3, 3}).cast(),
                                       SimplexArray(oSimplices.data(), {oSimplices.size() / 4, 4}).cast());
             })
        .def("load_surface",
             [](const std::string& path) {
                 GEO::Mesh mesh;
                 GEO::mesh_load(path, mesh);

                 const uint32_t dimension = mesh.vertices.dimension();

                 // Conversion
                 std::vector<GEO::index_t> simplices = {};
                 std::vector<double>       vertices  = {};
                 surfaceToArray(mesh, vertices, simplices);

                 return nb::make_tuple(VectorArray(vertices.data(), {vertices.size() / dimension, dimension}).cast(),
                                       SimplexArray(simplices.data(), {simplices.size() / 3, 3}).cast());
             })
        .def("load_volume",
             [](const std::string& path) {
                 GEO::Mesh mesh;
                 GEO::mesh_load(path, mesh);

                 const uint32_t dimension = mesh.vertices.dimension();
                 geo_assert(dimension == 3);

                 // Output
                 std::vector<double>       oVertices  = {};
                 std::vector<GEO::index_t> oSimplices = {};
                 volumeToArray(mesh, oVertices, oSimplices);

                 return nb::make_tuple(VectorArray(oVertices.data(), {oVertices.size() / dimension, dimension}).cast(),
                                       SimplexArray(oSimplices.data(), {oSimplices.size() / 4, 4}).cast());
             })
        .def("save", [](const VectorArray vertices, const SimplexArray simplices, const std::string& path) {
            const bool isTriangle = simplices.shape(1) == 3;
            const bool isTet      = simplices.shape(1) == 4;

            GEO::Mesh mesh;
            if (isTriangle)
                arrayToSurface({vertices.data(), vertices.size()}, {simplices.data(), simplices.size()}, mesh,
                               static_cast<uint32_t>(vertices.shape(1)));
            else if (isTet)
                arrayToVolume({vertices.data(), vertices.size()}, {simplices.data(), simplices.size()}, mesh);
            else
                geo_assert_not_reached;

            GEO::mesh_save(mesh, path);
        });
}
