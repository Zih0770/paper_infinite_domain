// Sample run:
// ./MeshingPhobos -out mesh/Phobos -h
// 30-300 -o 2 -buf 0.2 -alg 1 -alg2d 6 -Rref 11000 -data
// data/Phobos.out -decay_thickness 300

#include <gmsh.h>

#include "common.hpp"
#include "mfemElasticity.hpp"

using namespace mfemElasticity;

int main(int argc, char** argv) {
  try {
    Timer gtimer;
    double meshSizeMin = 50.0, meshSizeMax = 500.0;  // km
    double buffer_ratio = 0.2;                       // non-dim
    double topo_exag = 1.0;
    int elementOrder = 2, alg3d = 1, alg2d = 6, ignored_layers = 0;
    double Rref = 6371.0;
    double decay_thickness = 200;
    std::string premFileName = "data/prem.nocrust";
    std::string outputFileName = "mesh/prem_with_crust";
    std::string crustFile_d1 = "data/crust-1.0/crsthk.xyz";
    std::string crustFile_d2 = "data/crust-1.0/depthtomoho.xyz";

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-prem" && i + 1 < argc)
        premFileName = argv[++i];
      else if (arg == "-out" && i + 1 < argc)
        outputFileName = argv[++i];
      else if (arg == "-h" && i + 1 < argc) {
        auto ms = ParseDoubles(argv[++i]);
        if (ms.size() == 2) {
          meshSizeMin = ms[0];
          meshSizeMax = ms[1];
        } else {
          std::cerr << "Error: -h needs a-b\n";
          return 1;
        }
      } else if (arg == "-o" && i + 1 < argc)
        elementOrder = std::stoi(argv[++i]);
      else if (arg == "-buf" && i + 1 < argc)
        buffer_ratio = std::stod(argv[++i]);
      else if (arg == "-alg" && i + 1 < argc)
        alg3d = std::stoi(argv[++i]);
      else if (arg == "-alg2d" && i + 1 < argc)
        alg2d = std::stoi(argv[++i]);
      else if (arg == "-il" && i + 1 < argc)
        ignored_layers = std::stoi(argv[++i]);
      else if (arg == "-Rref" && i + 1 < argc)
        Rref = std::stod(argv[++i]);
      else if (arg == "-crust_d1" && i + 1 < argc)
        crustFile_d1 = argv[++i];
      else if (arg == "-crust_d2" && i + 1 < argc)
        crustFile_d2 = argv[++i];
      else if (arg == "-exag" && i + 1 < argc) {
        topo_exag = std::stod(argv[++i]);
        if (topo_exag < 0.0) {
          std::cerr << "Error: exag must be >= 0\n";
          return 1;
        }
      } else if (arg == "-decay_thickness" && i + 1 < argc)
        decay_thickness = std::stod(argv[++i]);
    }
    const double hmin = meshSizeMin / Rref;
    const double hmax = meshSizeMax / Rref;
    const double decay_thickness_nd = decay_thickness / Rref;

    // initialization
    gmsh::initialize();
    gmsh::model::add("Earth_PREM_CRUST");
    gtimer.Mark("Model created");

    // read PREM and construct layers
    PREMModel prem(premFileName, Rref * 1e3, buffer_ratio,
                   ignored_layers);  // PREM data is in meters
    auto& radii = prem.GetRadiiND();
    if (radii.size() < 3) {
      std::cerr << "Not enough PREM radii\n";
      gmsh::finalize();
      return 1;
    }
    std::cout << "Radii of " << radii.size()
              << " layers from PREM (without Moho&topo, with the outermost "
                 "boundary): ";
    for (double r : radii)
      std::cout << std::fixed << std::setprecision(8) << r << " ";
    std::cout << "(The length scale is " << std::fixed << std::setprecision(2)
              << Rref << " kilometers.)\n";
    gtimer.Mark("Read PREM and extracted layer boundaries");

    // two sets of topography data
    Topography topo_ext(crustFile_d1, Rref);
    Topography moho(crustFile_d2, Rref);
    Topography topo = topo_ext + moho;
    gtimer.Mark("Loaded CRUST-1.0 grids");

    // OCC and fragment
    double r_pre_topo = 1.0 + topo.Mean();
    double r_pre_moho = 1.0 + moho.Mean();
    std::cout << std::fixed << std::setprecision(6)
              << "[INFO] Average topo radius  (non-dim) = " << r_pre_topo
              << "  (" << (r_pre_topo * Rref) << " km)\n"
              << "[INFO] Average moho radius  (non-dim) = " << r_pre_moho
              << "  (" << (r_pre_moho * Rref) << " km)\n";
    std::vector<double> radii_ext = radii;
    radii_ext.insert(radii_ext.end(), {r_pre_moho, r_pre_topo});
    std::sort(radii_ext.begin(), radii_ext.end());

    int outerVol = gmsh::model::occ::addSphere(0., 0., 0., radii_ext.back());
    std::vector<std::pair<int, int>> objects = {{3, outerVol}};
    std::vector<std::pair<int, int>> tools;
    tools.reserve(radii_ext.size() - 1);
    for (std::size_t i = 0; i < radii_ext.size() - 1; ++i)
      tools.emplace_back(3,
                         gmsh::model::occ::addSphere(0., 0., 0., radii_ext[i]));
    gmsh::model::occ::synchronize();

    std::vector<std::pair<int, int>> outDimTags;
    std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;
    gmsh::model::occ::fragment(objects, tools, outDimTags, outDimTagsMap, -1,
                               true, true);
    gmsh::model::occ::removeAllDuplicates();
    gmsh::model::occ::synchronize();

    // Mesh size control
    std::vector<int> surfTags, volTags;
    for (const auto& p : outDimTags) {
      if (p.first == 3)
        volTags.push_back(p.second);
      else if (p.first == 2)
        surfTags.push_back(p.second);
    }
    std::sort(surfTags.begin(), surfTags.end());
    surfTags.erase(std::unique(surfTags.begin(), surfTags.end()),
                   surfTags.end());

    std::vector<double> facesListD(surfTags.begin(), surfTags.end());
    const double fac = 10.0, distMin = 0.0, distMax = fac * hmin;
    int fDist = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(fDist, "FacesList", facesListD);
    int fTh = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(fTh, "InField", fDist);
    gmsh::model::mesh::field::setNumber(fTh, "SizeMin", hmin);
    gmsh::model::mesh::field::setNumber(fTh, "SizeMax", hmax);
    gmsh::model::mesh::field::setNumber(fTh, "DistMin", distMin);
    gmsh::model::mesh::field::setNumber(fTh, "DistMax", distMax);
    gmsh::model::mesh::field::setAsBackgroundMesh(fTh);

    // Meshing
    gmsh::option::setNumber("Mesh.MeshSizeMin", hmin);
    gmsh::option::setNumber("Mesh.MeshSizeMax", hmax);
    // gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
    // gmsh::option::setNumber("Mesh.HighOrderOptimize", 1);
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    gmsh::option::setNumber("Mesh.Algorithm", alg2d);
    gmsh::option::setNumber("Mesh.Algorithm3D", alg3d);
    gmsh::option::setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0);

    gmsh::model::mesh::generate(3);
    gtimer.Mark("3D mesh generated");

    // perturb the nodes

    SpheroidalRadialSurface innerSurface(r_pre_moho);
    SpheroidalRadialSurface outerSurface(r_pre_topo);
    std::vector<const RadialSurface*> baseSurfaces = {&innerSurface,
                                                      &outerSurface};
    std::vector<const Topography*> topo_fields = {&moho, &topo};

    CubicBandLinearDecay mapping(topo_fields, baseSurfaces, decay_thickness_nd,
                                 topo_exag, 0, 1);

    gtimer.Mark("Applying node perturbation...");
    PerturbAllNodes(mapping);
    gtimer.Mark("Perturbation complete");

    // tagging
    TagLayersByRadius(volTags);
    gtimer.Mark("Tagged all volumes and surfaces (inner->outer = 1..N)");

    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName + ".msh");
    std::cout << "Wrote " << outputFileName << ".msh\n";
    gtimer.Mark("Mesh written to disk");

    gmsh::finalize();
    gtimer.Mark("Finalized Gmsh (done)");

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    gmsh::finalize();
    return 1;
  }
}
