#include <gmsh.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>


std::vector<double> extractLayerBoundaries(const std::string &fileName, double& R) {
    std::vector<double> radii;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << fileName << std::endl;
        return radii;
    }

    std::string line;
    double previousRadius = -1.0;

    int lineCount = 0;
    while (std::getline(file, line)) {
        if (lineCount < 3) {
            lineCount++;
            continue;
        }

        std::istringstream iss(line);
        double radius, density, pWave, sWave, bulkM, shearM;
        if (iss >> radius >> density >> pWave >> sWave >> bulkM >> shearM) {
            if (std::abs(radius - previousRadius) < 1e-3) {
                radii.push_back(radius);
            }
            previousRadius = radius;
        }
    }
    radii.push_back(previousRadius);
    if (R < 0) R = radii.back();
    double fac = 1.2;
    double radius_max = fac * R;
    radii.push_back(radius_max);
    for (double& r : radii) {
        r /= R;
    }

    file.close();
    return radii;
}

struct PropertyData {
    double radius;
    double density;
    double pWaveSpeed;
    double sWaveSpeed;
    double bm;
    double sm;
};

std::vector<std::vector<PropertyData>> parsePropertyData(const std::string &fileName, const std::vector<double> &radii, const double &R) {
    std::vector<std::vector<PropertyData>> data(radii.size());
    std::ifstream file(fileName);
    int attr = 0;

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << fileName << std::endl;
        return data;
    }

    std::string line;
    int lineCount = 0;

    while (std::getline(file, line)) {
        if (lineCount < 3) {
            lineCount++;
            continue;
        }

        std::istringstream iss(line);
        double radius, density, pWaveSpeed, sWaveSpeed, bulkQuality, shearQuality;
        if (iss >> radius >> density >> pWaveSpeed >> sWaveSpeed >> bulkQuality >> shearQuality) {
            data[attr].push_back({
                radius / R,
                density,
                pWaveSpeed,
                sWaveSpeed,
                bulkQuality,
                shearQuality
            });

            if (radius / R == radii[attr]) attr++; //
        }
    }

    file.close();
    return data;
}

double interpolateProperty(double r, const std::vector<PropertyData> &data, const std::function<double(const PropertyData&)> &propertyExtractor) {
    if (r < data.front().radius || r > data.back().radius) {
        std::cerr << "Warning: r=" << r << " out of range [" << data.front().radius << ", " << data.back().radius << "]\n";
        return 0.0;
    }

    for (int i = 0; i < data.size() - 1; ++i) {
        if (r >= data[i].radius && r <= data[i+1].radius) {
            double r1 = data[i].radius, r2 = data[i+1].radius;
            double p1 = propertyExtractor(data[i]), p2 = propertyExtractor(data[i+1]);
            return p1 + (p2 - p1) * (r - r1) / (r2 - r1);
        }
    }

    std::cout<<"Undefined behaviour."<<std::endl;
    return 0.0; //
}



void createConcentricSphericalLayers(const std::vector<double> &radii, const std::vector<std::vector<PropertyData>> &propertyDataList, double meshSizeMin, double meshSizeMax, int elementOrder, int algorithm, const std::string &outputFileName) {
    int numLayers = radii.size();
    if (numLayers < 1) {
        std::cerr << "Error: There should be at least one layer." << std::endl;
        return;
    }
    // Initialize Gmsh
    gmsh::initialize();
    gmsh::model::add("ConcentricSphericalLayers");

    // Set mesh size options
    gmsh::option::setNumber("Mesh.MeshSizeMin", meshSizeMin);
    gmsh::option::setNumber("Mesh.MeshSizeMax", meshSizeMax);

    int layerTag = 1;
    int surfaceTag = 1;
    for (int i = 0; i < numLayers; ++i) {
        gmsh::model::occ::addSphere(0, 0, 0, radii[i]);
    } 
    gmsh::model::occ::synchronize();
    gmsh::model::addPhysicalGroup(3, {1}, layerTag);
    gmsh::model::setPhysicalName(3, layerTag, "layer_1");
    std::vector<std::pair<int, int>> surfaceEntities;
    gmsh::model::getBoundary({{3, 1}}, surfaceEntities, false, false, false);
    std::pair<int, int> surface = surfaceEntities[0];
    if (surface.first == 2) {
        gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
        gmsh::model::setPhysicalName(2, surfaceTag, "surface_1");
    }
    for (int i = 1; i < numLayers; ++i) {
        std::vector<std::pair<int, int> > ov;
        std::vector<std::vector<std::pair<int, int> > > ovv;

        gmsh::model::occ::cut({{3, i+1}}, {{3, i}},  ov, ovv, -1, false, false); 
        gmsh::model::occ::synchronize();

        std::vector<int> volumeTags;
        for (const auto &entity : ov) {
            volumeTags.push_back(entity.second);  // Extract only the tag part
        }
        ++layerTag;
        gmsh::model::addPhysicalGroup(3, volumeTags, layerTag);
        gmsh::model::setPhysicalName(3, layerTag, "layer_" + std::to_string(i+1));
        for (const auto &volumeTag : volumeTags) {
            std::vector<std::pair<int, int>> surfaceEntities;
            gmsh::model::getBoundary({{3, volumeTag}}, surfaceEntities, false, false, false);	
            std::pair<int, int> surface = surfaceEntities[0];
            if (surface.first == 2) {
                ++surfaceTag;
                gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
                gmsh::model::setPhysicalName(2, surfaceTag, "surface_" + std::to_string(i+1));
            }
        }
    }
    for (int i = 1; i < numLayers; ++i) {
        gmsh::model::occ::remove({{3, i+1}});
    }
    gmsh::model::occ::synchronize();

    std::vector<double> facesList(numLayers);
    for (int i = 0; i < numLayers; ++i)
        facesList[i] = i + 1;

    gmsh::model::mesh::field::add("Distance", 1);
    gmsh::model::mesh::field::setNumbers(1, "FacesList", facesList);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", meshSizeMin); 
    gmsh::model::mesh::field::setNumber(2, "SizeMax", meshSizeMax);  
    gmsh::model::mesh::field::setNumber(2, "DistMin", 0.0);
    double fac = 10.0;
    gmsh::model::mesh::field::setNumber(2, "DistMax", meshSizeMin * fac);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    // Generate 3D mesh
    gmsh::option::setNumber("Mesh.Algorithm3D", algorithm); //1-Delaunay, 4-Frontal, 7-MMG3D, 9-R-tree Delaunay, 10-HXT (Frontal-Delaunay), 11-Automatic
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    //gmsh::option::setNumber("Mesh.HighOrderOptimize", 1);
    //gmsh::option::setNumber("Mesh.Optimize", 3);
    //gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);
    //gmsh::option::setNumber("Mesh.SecondOrderIncomplete", 0);
    //gmsh::option::setNumber("Mesh.Binary", 1);
    gmsh::model::mesh::generate(3);

    // Save the mesh
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName + ".msh");

    // Save the property parameters
    std::vector<size_t> nodeTags;
    std::vector<double> nodeCoords;
    std::vector<std::vector<double>> densityField, pWaveField, sWaveField, bmField, smField;

    std::vector<double> parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, parametricCoord, -1, -1, false, true);

    for (int ii = 0; ii < propertyDataList.size(); ++ii)
        std::cout<<"Size of "<<ii<<"th property block: "<<propertyDataList[ii].size()<<"."<<std::endl;

    std::ofstream gfFile(outputFileName + "_density.gf");
    gfFile << "FiniteElementSpace\n";
    gfFile << "FiniteElementCollection: L2_3D_P1\n";
    gfFile << "VDim: 1\n";
    gfFile << "Ordering: 0\n\n";  // Ordering = 0: by nodes

    for (const auto& valVec : densityField) {
        gfFile << std::setprecision(14) << valVec[0] << "\n";
    }

    for (int i = 0; i < nodeTags.size(); ++i) {
        double x = nodeCoords[3 * i];
        double y = nodeCoords[3 * i + 1];
        double z = nodeCoords[3 * i + 2];
        double r = std::sqrt(x * x + y * y + z * z);  // Radial distance

        int attr = -1;
        for (int j = 0; j < radii.size(); ++j) {
            if (r <= radii[j]) { //Shared nodes?
                attr = j;
                break;
            }
        }

        if (attr == -1 || attr == radii.size() - 1) {
            densityField.push_back({0.0});
            bmField.push_back({0.0});
            smField.push_back({0.0});
        } else {
            auto Density = std::vector<double>{
                interpolateProperty(r, propertyDataList[attr], [](const PropertyData &d) { return d.density; }) //
            };
            auto BM = std::vector<double>{
                interpolateProperty(r, propertyDataList[attr], [](const PropertyData &d) { return d.bm; })
            };
            auto SM = std::vector<double>{
                interpolateProperty(r, propertyDataList[attr], [](const PropertyData &d) { return d.sm; })
            };
            densityField.push_back(Density);
            bmField.push_back(BM);
            smField.push_back(SM);
        }
    }

    std::cout << "Size of nodeTags: " << nodeTags.size() << std::endl;
    std::cout << "Size of densityField: " << densityField.size() << std::endl;

    //std::vector<std::vector<double>> densityFieldWrapped(1, densityField);

    gmsh::view::add("Density");
    gmsh::view::addModelData(1, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, densityField);
    gmsh::view::write(1, outputFileName + "_density.pos");

    //gmsh::viewview::add("P-Wave Speed");
    //gmsh::view::addModelData(1, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, {pWaveField});
    //gmsh::view::write(1, "pwave_distribution.pos");

    //gmsh::view::add("S-Wave Speed");
    //gmsh::view::addModelData(2, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, {sWaveField});
    //gmsh::view::write(2, "swave_distribution.pos");

    gmsh::view::add("Bulk Modulus");
    gmsh::view::addModelData(2, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, bmField);
    gmsh::view::write(2, outputFileName + "_bm.pos");

    gmsh::view::add("Shear Modulus");
    gmsh::view::addModelData(3, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, smField);
    gmsh::view::write(3, outputFileName + "_sm.pos");

    for (const auto& valVec : densityField) {
        gfFile << std::setprecision(14) << valVec[0] << "\n";
    }
    gfFile.close();
    std::cout << "Density field saved to " << outputFileName + "_density.gf" << std::endl;

    // Finalize Gmsh
    gmsh::finalize();
}

std::vector<double> parseRadii(const std::string &radiiStr) {
    std::vector<double> radii;
    std::istringstream iss(radiiStr);
    std::string token;

    while (std::getline(iss, token, '-')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        radii.push_back(std::stod(token));
    }

    return radii;
}

int main(int argc, char **argv) {
    double meshSizeMin = 10e3;
    double meshSizeMax = 100e3;
    int algorithm = 1;
    int elementOrder = 2;
    std::string inputFileName = "data/prem.200.noiso";
    std::string outputFileName = "mesh/prem";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            inputFileName = argv[++i];
        } else if (arg == "-s" && i + 1 < argc) {
            std::string meshSizeStr = argv[++i];
            auto meshSizes = parseRadii(meshSizeStr); 
            if (meshSizes.size() == 2) {
                meshSizeMin = meshSizes[0];
                meshSizeMax = meshSizes[1];
            } else {
                std::cerr << "Error: mesh sizes should have two values.\n";
                return 1;
            }
        } else if (arg == "-o" && i + 1 < argc) {
            outputFileName = argv[++i];
        } else if (arg == "-order" && i + 1 < argc) {
            elementOrder = std::stoi(argv[++i]);
        } else if (arg == "-ma" && i + 1 < argc) {
            algorithm = std::stod(argv[++i]);
        }
    }

    //double R = -1.0;
    double R = 6371e3;
    std::vector<double> radii = extractLayerBoundaries(inputFileName, R);
    meshSizeMin /= R;
    meshSizeMax /= R;

    std::cout << "Detected radii of "<<radii.size()<<" layers: ";
    for (const double r : radii) {
        std::cout << std::fixed << std::setprecision(8) << r << " ";
    }
    std::cout << std::fixed << std::setprecision(2) << "(The length scale is "<<R<<" meters.)"<<std::endl;

    std::vector<std::vector<PropertyData>> propertyDataList = parsePropertyData(inputFileName, radii, R);

    // Run the spherical layers creation function
    createConcentricSphericalLayers(radii, propertyDataList, meshSizeMin, meshSizeMax, elementOrder, algorithm, outputFileName);

    return 0;
}

