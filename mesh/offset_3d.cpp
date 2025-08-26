#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <gmsh.h>
#include "common.hpp"

struct Params {
  double a = 0.25;   // small sphere radius
  double b = 1.0;    // big   sphere radius
  double t = 0.5;    // fraction of b for small-center offset
  double th_deg = 45.0; // angle in degrees in x–z plane from +x
  double small = 0.03, big = 0.30, fac = 0.50;
  double R = -1.0;   // outer radius; if <0, set to 1.2*b
  // centers (computed): small at (x0,y0,z0), big at origin
  double x0 = 0.0, y0 = 0.0, z0 = 0.0;
  double x1 = 0.0, y1 = 0.0, z1 = 0.0;
} P;

static void getd(int argc, char** argv, const char* key, double& dst){
  std::string k(key), eq = k + "=";
  for(int i=1;i<argc;++i){
    std::string s(argv[i]);
    if(s==k){ if(i+1<argc) dst = std::strtod(argv[++i], nullptr); }
    else if(s.rfind(eq,0)==0){ dst = std::strtod(s.c_str()+eq.size(), nullptr); }
  }
}
static bool hasKey(int argc, char** argv, const char* key){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){ std::string s(argv[i]); if(s==k || s.rfind(eq,0)==0) return true; }
  return false;
}
static void parseArgs(int argc, char** argv){
  getd(argc,argv,"-a",P.a);       getd(argc,argv,"-b",P.b);
  getd(argc,argv,"-t",P.t);       getd(argc,argv,"-th",P.th_deg);
  getd(argc,argv,"-small",P.small); getd(argc,argv,"-big",P.big);
  getd(argc,argv,"-fac",P.fac);   getd(argc,argv,"-R",P.R);

  if(P.R <= 0.0) P.R = 1.2 * P.b;
  const double th = P.th_deg * std::numbers::pi / 180.0;
  double d = P.t * P.b;
  if(d + P.a >= P.b){
    P.t = std::max(0.0, (P.b - P.a) / P.b * 0.95);
    d = P.t * P.b;
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
    "-a","-b","-t","-th","-small","-big","-fac","-R"
  };
  auto isOurs=[&](const std::string& s,std::string& k)->bool{
    for(const auto& key:ours) if(s==key || s.rfind(key+"=",0)==0){ k=key; return true; }
    return false;
  };
  argsStr.clear(); argsStr.reserve(argc); argsStr.emplace_back(argv[0]);
  for(int i=1;i<argc;++i){ std::string s(argv[i]),k;
    if(isOurs(s,k)){ if(s==k && i+1<argc) ++i; } else argsStr.emplace_back(std::move(s));
  }
  argvOut.clear(); argvOut.reserve(argsStr.size());
  for(auto& t:argsStr) argvOut.push_back(const_cast<char*>(t.c_str()));
}

static double meshSizeCallback(int, int, double x, double y, double z, double lc){
  const double rs = std::sqrt((x-P.x0)*(x-P.x0) + (y-P.y0)*(y-P.y0) + (z-P.z0)*(z-P.z0)); // dist to small center
  const double rb = std::sqrt((x-P.x1)*(x-P.x1) + (y-P.y1)*(y-P.y1) + (z-P.z1)*(z-P.z1)); // dist to big   center
  const double a = P.a, b = P.b, R = P.R, fac = P.fac;

  auto lerp = [](double A,double B,double t){ return A + (B - A)*t; };
  auto c01  = [](double t){ return t<0?0.0:(t>1?1.0:t); };

  const double ab = (a / b);
  const double dS = fac * a; // small boundary band thickness
  const double dB = fac * b; // big boundary band thickness

  if(rs < a){
    const double d = a - rs; // distance to small boundary (inside)
    if(d <= dS){
      const double t = c01(d / dS);
      return lerp(P.small*ab, P.big*ab, t);   // small*(a/b) -> big*(a/b)
    } else {
      return P.big*ab;                      
    }
  }

  if(rb < b){
    // near small boundary (outside side)
    const double d_outS = rs - a;
    if(d_outS >= 0.0 && d_outS <= dS){
      const double t = c01(d_outS / dS);
      return lerp(P.small*ab, P.big, t);      // small*(a/b) -> big
    }
    // near big boundary (inside side)
    const double d_inB = b - rb;
    if(d_inB >= 0.0 && d_inB <= dB){
      const double t = c01(d_inB / dB);
      return lerp(P.small, P.big, t);         // small -> big
    }
    return P.big;                              // elsewhere inside big
  }

  if(rb <= R){
    const double thick = std::max(R - b, 1e-12);
    const double slope = (P.big - P.small) / (dB > 0 ? dB : 1e-12); // (big-small)/(fac*b)
    const double d = rb - b;
    double h = P.small + slope * d;           // start at small at r=b
    const double h_end = P.small + slope * thick;
    if(h > h_end) h = h_end;                  // clamp at r=R
    return h;
  }

  // outside R (not meshed)
  (void)lc; return P.big;
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

  gmsh::model::add("spherical_offset_with_buffer");
  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  const double lc = 0.1;
  auto [sl1, s1] = createSphere(P.x0, P.y0, P.z0, P.a, lc); // small
  auto [sl2, s2] = createSphere(P.x1, P.y1, P.z1, P.b, lc); // big
  auto [sl3, s3] = createSphere(P.x1, P.y1, P.z1, P.R, lc); // outer

  int v_small = gmsh::model::geo::addVolume({sl1});
  int v_ring  = gmsh::model::geo::addVolume({sl2, sl1});
  int v_buf   = gmsh::model::geo::addVolume({sl3, sl2});
  gmsh::model::geo::synchronize();

  gmsh::model::addPhysicalGroup(3, {v_small}, 1);
  gmsh::model::addPhysicalGroup(3, {v_ring},  2);
  gmsh::model::addPhysicalGroup(3, {v_buf},   3);
  gmsh::model::addPhysicalGroup(2, s1, 11);
  gmsh::model::addPhysicalGroup(2, s2, 12);
  gmsh::model::addPhysicalGroup(2, s3, 13);

  gmsh::option::setNumber("Mesh.ElementOrder", 2);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::mesh::generate(3);
  gmsh::write("mesh/spherical_offset.msh");

  gmsh::option::setNumber("Mesh.Points", 0);
  gmsh::option::setNumber("Mesh.SurfaceEdges", 0);
  gmsh::option::setNumber("Mesh.SurfaceFaces", 1);

  bool no_popup=false;
  for(int i=1;i<argc;++i) if(std::string(argv[i])=="-nopopup"){ no_popup=true; break; }
  if(!no_popup) gmsh::fltk::run();

  gmsh::finalize();
  return 0;
}

