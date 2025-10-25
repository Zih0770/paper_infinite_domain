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
  double a = 0.1, b = 1.0;
  double x0 = 0.5, y0 = 0.5;     
  double x1 = 0.0, y1 = 0.0;     
  double Rout = 1.2;            
  double small = 0.01, big = 0.10, fac = 0.30;
  double t = 0.5, theta_deg = 45.0; 
} P;

static void getd(int argc, char** argv, const char* key, double& dst){
  std::string k(key), eq=k+"=";
  for(int i=1;i<argc;++i){
    std::string s(argv[i]);
    if(s==k){ if(i+1<argc) dst=std::strtod(argv[++i],nullptr); }
    else if(s.rfind(eq,0)==0){ dst=std::strtod(s.c_str()+eq.size(),nullptr); }
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
  getd(argc,argv,"-t",P.t);   getd(argc,argv,"-th",P.theta_deg);
  getd(argc,argv,"-R",P.Rout);
  getd(argc,argv,"-small",P.small); getd(argc,argv,"-big",P.big); getd(argc,argv,"-fac",P.fac);

  if(!hasKey(argc,argv,"-x0") && !hasKey(argc,argv,"-y0")){
    const double th=P.theta_deg*std::numbers::pi/180.0;
    double d=P.t*P.b;
    if(d+P.a>=P.b){ P.t=std::max(0.0,(P.b-P.a)/P.b*0.95); d=P.t*P.b; }
    P.x0=P.x1+d*std::cos(th); P.y0=P.y1+d*std::sin(th);
  }
}

static void buildFilteredArgsForGmsh(int argc, char** argv,
                                     std::vector<std::string>& argsStr,
                                     std::vector<char*>& argvOut){
  static const std::vector<std::string> ours={
    "-a","-b","-R","-t","-th","-x0","-y0","-x1","-y1","-small","-big","-fac"
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

static double meshSizeCallback(int, int, double x, double y, double z, double lc){
  using std::hypot;
  const double rs = hypot(x-P.x0, y-P.y0); // dist to small center
  const double rb = hypot(x-P.x1, y-P.y1); // dist to big center
  const double a = P.a, b = P.b, R = P.Rout, fac = P.fac;

  auto lerp = [](double A,double B,double t){ return A + (B - A)*t; };
  auto c01  = [](double t){ return t<0?0.0:(t>1?1.0:t); };

  const double ab = (a / b);
  const double dS = fac * a;   // band thickness around small boundary
  const double dB = fac * b;   // band thickness around big boundary

  // SMALL CIRCLE: rs < a
  if(rs < a){
    const double d = a - rs;                       // distance to small boundary (inside)
    if(d <= dS){
      const double t = c01(d / dS);
      return lerp(P.small*ab, P.big*ab, t);        // small*(a/b) -> big*(a/b)
    } else {
      return P.big*ab;                             // core constant
    }
  }

  // BIG CIRCLE (outside small): rb < b
  if(rb < b){
    // next to small boundary (outside side)
    const double d_outS = rs - a;
    if(d_outS >= 0.0 && d_outS <= dS){
      const double t = c01(d_outS / dS);
      return lerp(P.small*ab, P.big, t);           // small*(a/b) -> big
    }
    // next to big boundary (interior side)
    const double d_inB = b - rb;
    if(d_inB >= 0.0 && d_inB <= dB){
      const double t = c01(d_inB / dB);
      return lerp(P.small, P.big, t);              // small -> big
    }
    return P.big;                                   // elsewhere inside big
  }

  // BUFFER: b <= rb <= R
  if(rb <= R){
    // slope s = (big - small) / (fac * b)
    const double thick = std::max(R - b, 1e-12);
    const double s = (P.big - P.small) / (dB > 0 ? dB : 1e-12);
    const double d_outB = rb - b;                  // distance from big boundary (outside)
    double h = P.small + s * d_outB;
    const double h_end = P.small + s * thick;
    if(h > h_end) h = h_end;
    return h;
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

  gmsh::model::add("circular_offset_with_buffer");
  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  const double lc=0.1;
  auto [l1,b1]=createCircle(P.x0,P.y0,P.a,lc);
  auto [l2,b2]=createCircle(P.x1,P.y1,P.b,lc);
  auto [l3,b3]=createCircle(P.x1,P.y1,P.Rout,lc);

  int v_small=gmsh::model::geo::addPlaneSurface({l1});
  int v_ring =gmsh::model::geo::addPlaneSurface({l2,l1});
  int v_buf  =gmsh::model::geo::addPlaneSurface({l3,l2});
  gmsh::model::geo::synchronize();

  gmsh::model::addPhysicalGroup(2,{v_small},1);
  gmsh::model::addPhysicalGroup(2,{v_ring}, 2);
  gmsh::model::addPhysicalGroup(2,{v_buf},  3);
  gmsh::model::addPhysicalGroup(1,b1,11);
  gmsh::model::addPhysicalGroup(1,b2,12);
  gmsh::model::addPhysicalGroup(1,b3,13);

  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::mesh::generate(2);
  gmsh::write("mesh/circular_offset.msh");

  gmsh::option::setNumber("Mesh.Points", 0);
  gmsh::option::setNumber("Mesh.SurfaceEdges", 0);
  gmsh::option::setNumber("Mesh.SurfaceFaces", 1);

  bool no_popup=false;
  for(int i=1;i<argc;++i) if(std::string(argv[i])=="-nopopup"){ no_popup=true; break; }
  if(!no_popup) gmsh::fltk::run();

  gmsh::finalize();
  return 0;
}

