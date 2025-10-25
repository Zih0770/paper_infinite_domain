// simple_3d_two_spheres.cpp — small sphere embedded in a large sphere (3D)
// Shell sizing depends ONLY on distance from the small-sphere interface.
// New: -out <path> to choose output .msh location (creates directories if needed)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
#include <filesystem>  // NEW

#include <gmsh.h>
#include "common.hpp" // assumes: createSphere(x,y,z,r,lc) -> {surfaceLoopTag, std::vector<int> surfaceTags}

struct Params {
  double a = 0.25;                // small sphere radius
  double b = 1.0;                 // big   sphere radius
  double t = 0.5;                 // offset as fraction of b
  double th_deg = 45.0;           // angle (deg) in x–z plane from +x
  double small = 0.03, big = 0.30, fac = 0.50;

  double x0 = 0.0, y0 = 0.0, z0 = 0.0; // small center
  double x1 = 0.0, y1 = 0.0, z1 = 0.0; // big   center (origin)
  std::string out = "mesh/simple_3d.msh";      // NEW: output path
} P;

// --------- arg parsing ----------
static void getd(int argc, char** argv, const char* key, double& dst){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){
    std::string s(argv[i]);
    if(s==k){ if(i+1<argc) dst=std::strtod(argv[++i],nullptr); }
    else if(s.rfind(eq,0)==0){ dst=std::strtod(s.c_str()+eq.size(),nullptr); }
  }
}
static void gets(int argc, char** argv, const char* key, std::string& dst){ // NEW
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){
    std::string s(argv[i]);
    if(s==k){ if(i+1<argc) dst=argv[++i]; }
    else if(s.rfind(eq,0)==0){ dst=s.substr(eq.size()); }
  }
}
static bool hasKey(int argc, char** argv, const char* key){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){ std::string s(argv[i]); if(s==k||s.rfind(eq,0)==0) return true; }
  return false;
}
static void parseArgs(int argc, char** argv){
  getd(argc,argv,"-a",P.a);     getd(argc,argv,"-b",P.b);
  getd(argc,argv,"-t",P.t);     getd(argc,argv,"-th",P.th_deg);
  getd(argc,argv,"-small",P.small); getd(argc,argv,"-big",P.big); getd(argc,argv,"-fac",P.fac);
  gets(argc,argv,"-out",P.out); gets(argc,argv,"-o",P.out); // NEW

  // place small-sphere center at distance t*b from big center in x–z plane
  const double th = P.th_deg * std::numbers::pi / 180.0;
  double d = P.t * P.b;
  if(d + P.a >= P.b){
    P.t = std::max(0.0, (P.b - P.a) / P.b * 0.95);
    d   = P.t * P.b;
    std::cerr << "[info] adjusted -t to " << P.t << " so the small sphere fits inside the big sphere.\n";
  }
  P.x0 = P.x1 + d * std::cos(th);
  P.y0 = P.y1 + 0.0;
  P.z0 = P.z1 + d * std::sin(th);
}

static void buildFilteredArgsForGmsh(int argc, char** argv,
                                     std::vector<std::string>& argsStr,
                                     std::vector<char*>& argvOut){
  static const std::vector<std::string> ours={
    "-a","-b","-t","-th","-small","-big","-fac","-out","-o" // NEW keys
  };
  auto isOurs=[&](const std::string& s,std::string& k)->bool{
    for(const auto& key:ours) if(s==key||s.rfind(key+"=",0)==0){ k=key; return true; }
    return false;
  };
  argsStr.clear(); argsStr.reserve(argc); argsStr.emplace_back(argv[0]);
  for(int i=1;i<argc;++i){
    std::string s(argv[i]),k;
    if(isOurs(s,k)){ if(s==k && i+1<argc) ++i; } else argsStr.emplace_back(std::move(s));
  }
  argvOut.clear(); argvOut.reserve(argsStr.size());
  for(auto& t:argsStr) argvOut.push_back(const_cast<char*>(t.c_str()));
}

// --------- size callback (smooth, single-driver in shell) ----------
static double meshSizeCallback(int, int, double x, double y, double z, double /*lc*/){
  const double rs = std::sqrt((x-P.x0)*(x-P.x0) + (y-P.y0)*(y-P.y0) + (z-P.z0)*(z-P.z0)); // dist to small
  const double rb = std::sqrt((x-P.x1)*(x-P.x1) + (y-P.y1)*(y-P.y1) + (z-P.z1)*(z-P.z1)); // dist to big
  const double a = P.a, b = P.b, fac = P.fac;

  auto lerp    = [](double A,double B,double t){ return A + (B - A)*t; };
  auto clamp01 = [](double t){ return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); };

  const double eps = 1e-14;
  const double ratio_ab = (b > eps) ? (a / b) : 0.0; // a/b
  const double dS = fac * a;

  if(rb >= b) return P.big;

  if(rs < a){
    const double din = a - rs;
    if(din <= dS){
      const double t = clamp01(din / (dS > eps ? dS : eps));
      return lerp(P.small, ratio_ab * P.big, t);
    } else {
      return ratio_ab * P.big;
    }
  }

  const double doutS = rs - a;
  if(doutS <= dS){
    const double t = clamp01(doutS / (dS > eps ? dS : eps));
    return lerp(P.small, P.big, t);
  }
  return P.big;
}

int main(int argc, char** argv){
  parseArgs(argc, argv);

  std::vector<std::string> gmshArgsStr; std::vector<char*> gmshArgv;
  buildFilteredArgsForGmsh(argc, argv, gmshArgsStr, gmshArgv);
  gmsh::initialize((int)gmshArgv.size(), gmshArgv.data());

  gmsh::option::setNumber("General.Terminal", 1);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  gmsh::model::add("two_spheres");
  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  const double lc = 0.1;
  auto [sl_small, surf_small] = createSphere(P.x0, P.y0, P.z0, P.a, lc);
  auto [sl_big,   surf_big]   = createSphere(P.x1, P.y1, P.z1, P.b, lc);

  int v_small = gmsh::model::geo::addVolume({sl_small});
  int v_shell = gmsh::model::geo::addVolume({sl_big, sl_small});
  gmsh::model::geo::synchronize();

  gmsh::model::addPhysicalGroup(3, {v_small}, 1);
  gmsh::model::addPhysicalGroup(3, {v_shell}, 2);
  gmsh::model::addPhysicalGroup(2,  surf_small, 11);
  gmsh::model::addPhysicalGroup(2,  surf_big,   12);

  gmsh::option::setNumber("Mesh.ElementOrder", 2);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);
  gmsh::option::setNumber("Mesh.Algorithm3D", 4); // Delaunay

  gmsh::model::mesh::generate(3);

  // Ensure output directory exists (NEW)
  std::filesystem::path outPath(P.out);
  if(outPath.has_parent_path()) std::filesystem::create_directories(outPath.parent_path());
  gmsh::write(P.out.c_str());

  gmsh::option::setNumber("Mesh.Points", 0);
  gmsh::option::setNumber("Mesh.SurfaceEdges", 0);
  gmsh::option::setNumber("Mesh.SurfaceFaces", 1);

  bool no_popup=false;
  for(int i=1;i<argc;++i) if(std::string(argv[i])=="-nopopup"){ no_popup=true; break; }
  if(!no_popup) gmsh::fltk::run();

  gmsh::finalize();
  return 0;
}

