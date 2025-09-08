#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
#include <filesystem>  // keep if you already support -out

#include <gmsh.h>
#include "common.hpp" // assumes createSphere(...) is available

struct Params {
  double a = 1.0;                 // small sphere radius (default like 2D)
  double b = 4.0;                 // big sphere radius   (default like 2D)
  double d = 2.0;                 // offset distance from big center (default like 2D: 0.5*b)
  double th_deg = 45.0;           // angle (deg) in x–z plane from +x
  double small = 0.05, big = 0.50, fac = 0.50; // defaults analogous to your 2D example
  double x0 = 0.0, y0 = 0.0, z0 = 0.0; // small center (computed)
  double x1 = 0.0, y1 = 0.0, z1 = 0.0; // big center (origin)
  std::string out = "mesh/simple_3d.msh"; // default output
} P;

static void getd(int argc, char** argv, const char* key, double& dst){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){ std::string s(argv[i]);
    if(s==k){ if(i+1<argc) dst=std::strtod(argv[++i],nullptr); }
    else if(s.rfind(eq,0)==0){ dst=std::strtod(s.c_str()+eq.size(),nullptr); }
  }
}
static void gets(int argc, char** argv, const char* key, std::string& dst){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){ std::string s(argv[i]);
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
  getd(argc,argv,"-d",P.d);     getd(argc,argv,"-th",P.th_deg);
  getd(argc,argv,"-small",P.small); getd(argc,argv,"-big",P.big); getd(argc,argv,"-fac",P.fac);
  gets(argc,argv,"-out",P.out); gets(argc,argv,"-o",P.out); // keep existing option

  // place small-sphere center at distance d from big center in x–z plane
  const double th = P.th_deg * std::numbers::pi / 180.0;
  double d = hasKey(argc,argv,"-d") ? P.d : (0.5 * P.b); // fallback matches old t=0.5 if -d omitted
  if(d + P.a >= P.b){ d = std::max(0.0, (P.b - P.a) * 0.95); }
  P.x0 = P.x1 + d * std::cos(th);
  P.y0 = P.y1 + 0.0;
  P.z0 = P.z1 + d * std::sin(th);
}
static void buildFilteredArgsForGmsh(int argc, char** argv,
                                     std::vector<std::string>& argsStr,
                                     std::vector<char*>& argvOut){
  static const std::vector<std::string> ours={
    "-a","-b","-d","-th","-small","-big","-fac","-out","-o"
  };
  auto isOurs=[&](const std::string& s,std::string& k)->bool{
    for(const auto& key:ours) if(s==key||s.rfind(key+"=",0)==0){ k=key; return true; }
    return false;
  };
  argsStr.clear(); argsStr.reserve(argc); argsStr.emplace_back(argv[0]);
  for(int i=1;i<argc;++i){ std::string s(argv[i]),k;
    if(isOurs(s,k)){ if(s==k && i+1<argc) ++i; } else argsStr.emplace_back(std::move(s));
  }
  argvOut.clear(); argvOut.reserve(argsStr.size());
  for(auto& t:argsStr) argvOut.push_back(const_cast<char*>(t.c_str()));
}

// --- size callback: two simultaneous boundary-layer ramps in the shell ---
static double meshSizeCallback(int, int, double x, double y, double z, double /*lc*/){
  const double rs = std::sqrt((x-P.x0)*(x-P.x0) + (y-P.y0)*(y-P.y0) + (z-P.z0)*(z-P.z0));
  const double rb = std::sqrt((x-P.x1)*(x-P.x1) + (y-P.y1)*(y-P.y1) + (z-P.z1)*(z-P.z1));
  const double a = P.a, b = P.b, fac = P.fac;

  auto lerp    = [](double A,double B,double t){ return A + (B - A)*t; };
  auto clamp01 = [](double t){ return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); };

  const double eps = 1e-14;
  const double ratio_ab = (b > eps) ? (a / b) : 0.0;                      // a/b
  const double small_outer = (a > eps) ? (P.small * (b / a)) : P.small;    // small*b/a
  const double d_in  = fac * a;   // depth from r=a into shell
  const double d_out = fac * b;   // depth from r=b into interior

  if(rb >= b) return P.big;

  // Inside small sphere: ramp from small at r=a inward to (a/b)*big over fac*a
  if(rs < a){
    const double din = a - rs;
    if(din <= d_in){
      const double t = clamp01(din / (d_in > eps ? d_in : eps));
      return lerp(P.small, ratio_ab * P.big, t);
    } else {
      return ratio_ab * P.big;
    }
  }

  // Shell (a <= r <= b): two ramps active simultaneously; take the minimum
  // 1) From inner boundary: small -> big over depth fac*a
  double size_from_inner = P.big;
  {
    const double d = rs - a; // distance from inner boundary
    const double t = clamp01((d_in > eps) ? (d / d_in) : 1.0);
    size_from_inner = lerp(P.small, P.big, t);
  }
  // 2) From outer boundary: small*b/a -> big over depth fac*b
  double size_from_outer = P.big;
  {
    const double d = b - rb; // inward distance from outer boundary
    const double t = clamp01((d_out > eps) ? (d / d_out) : 1.0);
    size_from_outer = lerp(small_outer, P.big, t);
  }

  return std::min(size_from_inner, size_from_outer);
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
  gmsh::option::setNumber("Mesh.Algorithm3D", 4); // Delaunay (unchanged)

  gmsh::model::mesh::generate(3);

  // write (default)
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

