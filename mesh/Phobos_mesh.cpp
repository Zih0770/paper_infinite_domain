// Sample run:
// ./Phobos_mesh -out mesh/Phobos -h
// 0.1-0.5 -o 2 -buf 0.2 -alg 1 -alg2d 6 -Rref 11 -data
// data/PhobosRadius_reordered.out -decay_thickness 1

#include <gmsh.h>

#include "common.hpp"

int main(int argc, char** argv) {
  try {
    double meshSizeMin = 0.1, meshSizeMax = 0.5;  // km
    double buffer_ratio = 0.3;                       // non-dim
    int elementOrder = 2, alg3d = 1, alg2d = 6;
    double Rref = 11.0;
    double decay_thickness = 1;
    std::string outputFileName = "mesh/Phobos";
    std::string dataFile = "data/PhobosRadius_reordered.out";

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-out" && i + 1 < argc)
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
      else if (arg == "-Rref" && i + 1 < argc)
        Rref = std::stod(argv[++i]);
      else if (arg == "-data" && i + 1 < argc)
        dataFile = argv[++i];
      else if (arg == "-decay_thickness" && i + 1 < argc)
        decay_thickness = std::stod(argv[++i]);
    }
    const double hmin = meshSizeMin / Rref;
    const double hmax = meshSizeMax / Rref;
    const double decay_thickness_nd = decay_thickness / Rref;

    gmsh::initialize();
    gmsh::model::add("Phobos");

    double pre_topo = 1.0; 
    double outer_bdr = pre_topo * (1 + buffer_ratio);

    Topography topo(dataFile, Rref * 1e3);
    topo += -pre_topo;

    int outerVol = gmsh::model::occ::addSphere(0., 0., 0., outer_bdr);
    std::vector<std::pair<int, int>> objects = {{3, outerVol}};
    std::vector<std::pair<int, int>> tools;
    tools.emplace_back(3, gmsh::model::occ::addSphere(0., 0., 0., pre_topo));
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
    //gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
    //gmsh::option::setNumber("Mesh.HighOrderOptimize", 1);
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    gmsh::option::setNumber("Mesh.Algorithm", alg2d);
    gmsh::option::setNumber("Mesh.Algorithm3D", alg3d);
    gmsh::option::setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0);

    gmsh::model::mesh::generate(3);

    // perturb the nodes
    SpheroidalRadialSurface innerSurface(pre_topo);
    std::vector<const RadialSurface*> baseSurfaces = {&innerSurface};
    std::vector<const Topography*> topo_fields = {&topo};

    LinearDecay mapping(topo_fields, baseSurfaces, decay_thickness_nd);

    PerturbAllNodes(mapping);

    // tagging
    TagLayersByRadius(volTags);

    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName + ".msh");
    std::cout << "Wrote " << outputFileName << ".msh\n";

    gmsh::finalize();

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    gmsh::finalize();
    return 1;
  }
}
