#pragma once

#include <gmsh.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <vector>

// Simple struct for circles.
struct Circle {
  const double x0;
  const double y0;
  const double r;

  Circle(double x0, double y0, double r) : x0{x0}, y0{y0}, r{r} {}

  double DistanceTo(double x, double y) const {
    return std::hypot(x - x0, y - y0);
  }
};

class Circles {
 private:
  std::vector<Circle> _circles;
  std::vector<double> _small;

 public:
};

// Simple stuct for spheres.
struct Sphere {
  double x0;
  double y0;
  double z0;
  double r;

  Sphere(double x0, double y0, double z0, double r)
      : x0{x0}, y0{y0}, z0{z0}, r{r} {}

  double DistanceTo(double x, double y, double z) const {
    return std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) +
                     (z - z0) * (z - z0));
  }
};

// Helper function to create a circle geometry
// Returns a pair: first is the curve loop tag, second is a vector of curve tags
std::pair<int, std::vector<int>> createCircle(Circle&& circle, double size);

// Helper function to create a circle geometry
// Returns a pair: first is the curve loop tag, second is a vector of curve tags
std::pair<int, std::vector<int>> createCircle(double x, double y, double r,
                                              double lc_val);

// Helper function to create a sphere geometry
// Returns a pair: first is the surface loop tag, second is a vector of surface
// tags
std::pair<int, std::vector<int>> createSphere(double x, double y, double z,
                                              double r, double lc_val);
