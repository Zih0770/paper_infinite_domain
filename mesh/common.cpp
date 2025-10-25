#include "common.hpp"

std::pair<int, std::vector<int>> createCircle(Circle circle, double size) {
  int p1 = gmsh::model::geo::addPoint(circle.x0, circle.y0, 0, size);
  int p2 = gmsh::model::geo::addPoint(circle.x0 + circle.r, circle.y0, 0, size);
  int p3 = gmsh::model::geo::addPoint(circle.x0, circle.y0 + circle.r, 0, size);
  int p4 = gmsh::model::geo::addPoint(circle.x0 - circle.r, circle.y0, 0, size);
  int p5 = gmsh::model::geo::addPoint(circle.x0, circle.y0 - circle.r, 0, size);

  int c1 = gmsh::model::geo::addCircleArc(p2, p1, p3);
  int c2 = gmsh::model::geo::addCircleArc(p3, p1, p4);
  int c3 = gmsh::model::geo::addCircleArc(p4, p1, p5);
  int c4 = gmsh::model::geo::addCircleArc(p5, p1, p2);

  std::vector<int> curve_tags = {c1, c2, c3, c4};
  int curve_loop_tag = gmsh::model::geo::addCurveLoop(curve_tags);

  return {curve_loop_tag, curve_tags};
}

std::pair<int, std::vector<int>> createCircle(double x, double y, double r,
                                              double lc_val) {
  int p1 = gmsh::model::geo::addPoint(x, y, 0, lc_val);
  int p2 = gmsh::model::geo::addPoint(x + r, y, 0, lc_val);
  int p3 = gmsh::model::geo::addPoint(x, y + r, 0, lc_val);
  int p4 = gmsh::model::geo::addPoint(x - r, y, 0, lc_val);
  int p5 = gmsh::model::geo::addPoint(x, y - r, 0, lc_val);

  int c1 = gmsh::model::geo::addCircleArc(p2, p1, p3);
  int c2 = gmsh::model::geo::addCircleArc(p3, p1, p4);
  int c3 = gmsh::model::geo::addCircleArc(p4, p1, p5);
  int c4 = gmsh::model::geo::addCircleArc(p5, p1, p2);

  std::vector<int> curve_tags = {c1, c2, c3, c4};
  int curve_loop_tag = gmsh::model::geo::addCurveLoop(curve_tags);

  return {curve_loop_tag, curve_tags};
}

std::pair<int, std::vector<int>> createSphere(double x, double y, double z,
                                              double r, double lc_val) {
  // Define points on the sphere and center
  int p1 = gmsh::model::geo::addPoint(x, y, z, lc_val);      // Center
  int p2 = gmsh::model::geo::addPoint(x + r, y, z, lc_val);  // +X
  int p3 = gmsh::model::geo::addPoint(x, y + r, z, lc_val);  // +Y
  int p4 = gmsh::model::geo::addPoint(x, y, z + r, lc_val);  // +Z
  int p5 = gmsh::model::geo::addPoint(x - r, y, z, lc_val);  // -X
  int p6 = gmsh::model::geo::addPoint(x, y - r, z, lc_val);  // -Y
  int p7 = gmsh::model::geo::addPoint(x, y, z - r, lc_val);  // -Z

  // Define circular arcs (edges of the sphere's "patches")
  int c1 = gmsh::model::geo::addCircleArc(p2, p1, p7);
  int c2 = gmsh::model::geo::addCircleArc(p7, p1, p5);
  int c3 = gmsh::model::geo::addCircleArc(p5, p1, p4);
  int c4 = gmsh::model::geo::addCircleArc(p4, p1, p2);
  int c5 = gmsh::model::geo::addCircleArc(p2, p1, p3);
  int c6 = gmsh::model::geo::addCircleArc(p3, p1, p5);
  int c7 = gmsh::model::geo::addCircleArc(p5, p1, p6);
  int c8 = gmsh::model::geo::addCircleArc(p6, p1, p2);
  int c9 = gmsh::model::geo::addCircleArc(p7, p1, p3);
  int c10 = gmsh::model::geo::addCircleArc(p3, p1, p4);
  int c11 = gmsh::model::geo::addCircleArc(p4, p1, p6);
  int c12 = gmsh::model::geo::addCircleArc(p6, p1, p7);

  // Define curve loops for each "patch" of the sphere
  int l1 = gmsh::model::geo::addCurveLoop({c5, c10, c4});
  int l2 = gmsh::model::geo::addCurveLoop({c9, -c5, c1});
  int l3 = gmsh::model::geo::addCurveLoop({c12, -c8, -c1});
  int l4 = gmsh::model::geo::addCurveLoop({c8, -c4, c11});
  int l5 = gmsh::model::geo::addCurveLoop({-c10, c6, c3});
  int l6 = gmsh::model::geo::addCurveLoop({-c11, -c3, c7});
  int l7 = gmsh::model::geo::addCurveLoop({-c2, -c7, -c12});
  int l8 = gmsh::model::geo::addCurveLoop({-c6, -c9, c2});

  // Create surfaces from the curve loops using SurfaceFilling
  int s1 = gmsh::model::geo::addSurfaceFilling({l1});
  int s2 = gmsh::model::geo::addSurfaceFilling({l2});
  int s3 = gmsh::model::geo::addSurfaceFilling({l3});
  int s4 = gmsh::model::geo::addSurfaceFilling({l4});
  int s5 = gmsh::model::geo::addSurfaceFilling({l5});
  int s6 = gmsh::model::geo::addSurfaceFilling({l6});
  int s7 = gmsh::model::geo::addSurfaceFilling({l7});
  int s8 = gmsh::model::geo::addSurfaceFilling({l8});

  std::vector<int> surface_tags = {s1, s2, s3, s4, s5, s6, s7, s8};
  int surface_loop_tag = gmsh::model::geo::addSurfaceLoop(surface_tags);

  return {surface_loop_tag, surface_tags};
}
