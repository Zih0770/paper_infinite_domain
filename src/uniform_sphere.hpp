#pragma once

#include <cmath>

#include "mfem.hpp"

class UniformSphereSolution {
 private:
  static constexpr mfem::real_t pi = std::atan(1) * 4;

  int _dim;
  mfem::real_t _r;

  mfem::Vector _x;

 public:
  UniformSphereSolution(int dim, const mfem::Vector& x, mfem::real_t r)
      : _dim{dim}, _x{x}, _r{r} {}

  mfem::FunctionCoefficient Coefficient() const {
    using namespace mfem;
    if (_dim == 2) {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(_x);
        if (r <= _r) {
          return pi * r * r;
        } else {
          return 2 * pi * _r * log(r / _r) + pi * _r * _r;
        }
      });
    } else {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(_x);
        if (r <= _r) {
          return -2 * pi * (3 * _r * _r - r * r) / 3;
        } else {
          return -4 * pi * pow(_r, 3) / (3 * r);
        }
      });
    }
  }

  mfem::FunctionCoefficient LinearisedCoefficient(const mfem::Vector& a) const {
    using namespace mfem;

    if (_dim == 2) {
      return FunctionCoefficient([this, a](const Vector& x) {
        auto dim = x.Size();
        auto r = x.DistanceTo(_x);
        auto dr = x;
        dr -= _x;
        dr /= r;
        if (r <= _r) {
          return -2 * pi * r * (dr * a);
        } else {
          return -2 * pi * (dr * a) / r;
        }
      });
    } else {
      return FunctionCoefficient([this, &a](const Vector& x) {
        auto dim = x.Size();
        auto r = x.DistanceTo(_x);
        auto dr = x;
        dr -= _x;
        dr /= r;
        if (r <= _r) {
          return -4 * pi * r * (dr * a) / 3;
        } else {
          return -4 * pi * std::pow(_r, 3) * (dr * a) / (3 * r * r);
        }
      });
    }
  }
};
