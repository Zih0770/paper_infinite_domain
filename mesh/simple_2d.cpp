#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

#include <gmsh.h>
#include "common.hpp"

struct Params {
  double a = 0.1, b = 1.0;
  double x0 = 0.5, y0 = 0.5;      // small-circle center
  double x1 = 0.0, y1 = 0.0;      // big-circle center
  double small = 0.01, big = 0.10, fac = 0.30;
  double d = 0.0, theta_deg = 45.0;            // placement via distance d at angle theta
  std::string out = "mesh/simple_2d.msh";
  int elemOrder = 2;  
  int alg2D = 6;       // example default (Frontal-Delaunay)
} P;
static void checkParams() {
  if (P.a <= 0.0 || P.b <= 0.0 || P.a >= P.b) {
    std::cerr << "Error: require 0 < a < b (got a=" << P.a << ", b=" << P.b << ")\n";
    std::exit(EXIT_FAILURE);
  }
  if (P.fac <= 0.0) {
    std::cerr << "Error: fac must be > 0 (got " << P.fac << ")\n";
    std::exit(EXIT_FAILURE);
  }
  const double ratio_ab    = P.a / P.b;               // a/b
  const double small_outer = P.small * (P.b / P.a);   // small*b/a
  if (ratio_ab * P.big < P.small) {
    std::cerr << "Error: ratio_ab * big (" << ratio_ab * P.big
              << ") < small (" << P.small << ")\n";
    std::exit(EXIT_FAILURE);
  }
  if (small_outer > P.big) {
    std::cerr << "Error: small_outer (" << small_outer
              << ") > big (" << P.big << ")\n";
    std::exit(EXIT_FAILURE);
  }
}

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
static void geti(int argc, char** argv, const char* key, int& dst){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){
    std::string s(argv[i]);
    if(s==k){ if(i+1<argc) dst=std::atoi(argv[++i]); }
    else if(s.rfind(eq,0)==0){ dst=std::atoi(s.c_str()+eq.size()); }
  }
}
static bool hasKey(int argc, char** argv, const char* key){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){ std::string s(argv[i]); if(s==k||s.rfind(eq,0)==0) return true; }
  return false;
}

static void parseArgs(int argc, char** argv){
  getd(argc,argv,"-a",P.a);   getd(argc,argv,"-b",P.b);
  getd(argc,argv,"-x0",P.x0); getd(argc,argv,"-y0",P.y0);
  getd(argc,argv,"-x1",P.x1); getd(argc,argv,"-y1",P.y1);
  getd(argc,argv,"-d",P.d);   getd(argc,argv,"-th",P.theta_deg);
  getd(argc,argv,"-small",P.small); getd(argc,argv,"-big",P.big); getd(argc,argv,"-fac",P.fac);
  gets(argc,argv,"-out",P.out); gets(argc,argv,"-o",P.out);
  geti(argc,argv,"-order",P.elemOrder);
  geti(argc,argv,"-alg2d",P.alg2D);

  if(!hasKey(argc,argv,"-x0") && !hasKey(argc,argv,"-y0")){
    const double th=P.theta_deg*std::numbers::pi/180.0;
    double d = hasKey(argc,argv,"-d") ? P.d : (0.5 * P.b);
    if(d + P.a >= P.b){ d = std::max(0.0, (P.b - P.a)*0.95); }
    P.x0 = P.x1 + d*std::cos(th);
    P.y0 = P.y1 + d*std::sin(th);
  }
}

static void buildFilteredArgsForGmsh(int argc, char** argv,
                                     std::vector<std::string>& argsStr,
                                     std::vector<char*>& argvOut){
  // add the new flags here so they don't reach Gmsh
  static const std::vector<std::string> ours={
    "-a","-b","-d","-th","-x0","-y0","-x1","-y1","-small","-big","-fac","-out","-o",
    "-order","-alg2d","-nopopup"
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

static double meshSizeCallback(int, int, double x, double y, double /*z*/, double /*lc*/){
  using std::hypot;
  const double rs = hypot(x - P.x0, y - P.y0); // distance to small center
  const double rb = hypot(x - P.x1, y - P.y1); // distance to big center
  const double a = P.a, b = P.b, fac = P.fac;

  auto lerp    = [](double A,double B,double t){ return A + (B - A)*t; };
  auto clamp01 = [](double t){ return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); };

  const double ratio_ab    = a / b;                 // a/b
  const double small_outer = P.small * (b / a);     // small*b/a
  const double d_in  = fac * a;                     // depth from r=a
  const double d_out = fac * b;                     // depth from r=b

  if(rb > b) return small_outer;                        

  // inside small circle
  if(rs < a){
    const double din = a - rs; // >= 0
    if(din <= d_in){
      const double t = clamp01(din / d_in);
      return lerp(P.small, ratio_ab * P.big, t);
    } else {
      return ratio_ab * P.big;
    }
  }

  // in the annulus: take min of inner/outer ramps
  const double d_from_inner = rs - a;  // >= 0
  const double t_in  = clamp01(d_from_inner / d_in);
  const double size_from_inner = lerp(P.small, P.big, t_in);

  const double d_from_outer = b - rb;  // >= 0
  const double t_out = clamp01(d_from_outer / d_out);
  const double size_from_outer = lerp(small_outer, P.big, t_out);

  return std::min(size_from_inner, size_from_outer);
}

int main(int argc, char** argv){
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();

  parseArgs(argc, argv);
  checkParams();

  std::vector<std::string> gmshArgsStr; std::vector<char*> gmshArgv;
  buildFilteredArgsForGmsh(argc, argv, gmshArgsStr, gmshArgv);
  gmsh::initialize((int)gmshArgv.size(), gmshArgv.data());

  gmsh::option::setNumber("General.Terminal", 1);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  gmsh::model::add("circle_in_circle");
  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  const double lc = 0.1;
  auto [l1,b1] = createCircle(P.x0,P.y0,P.a,lc);
  auto [l2,b2] = createCircle(P.x1,P.y1,P.b,lc);

  int v_small = gmsh::model::geo::addPlaneSurface({l1});
  int v_ring  = gmsh::model::geo::addPlaneSurface({l2,l1});
  gmsh::model::geo::synchronize();

  gmsh::model::addPhysicalGroup(2,{v_small},1);
  gmsh::model::addPhysicalGroup(2,{v_ring}, 2);
  gmsh::model::addPhysicalGroup(1,b1,1);
  gmsh::model::addPhysicalGroup(1,b2,2);

  gmsh::option::setNumber("Mesh.ElementOrder", P.elemOrder);
  gmsh::option::setNumber("Mesh.Algorithm",    P.alg2D);

  gmsh::model::mesh::generate(2);

  const auto t1 = clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "2D meshing took " << seconds << " s\n";

  // write
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

