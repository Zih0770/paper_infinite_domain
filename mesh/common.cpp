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

void TagLayersByRadius(const std::vector<int>& volTags,
                       const std::string& volPrefix,
                       const std::string& surfPrefix) {
  gmsh::model::removePhysicalGroups();
  std::vector<std::tuple<int, int, double>>
      layers;  // (vol, outerSurf, r_outer)
  layers.reserve(volTags.size());

  for (int v : volTags) {
    gmsh::vectorpair bnd;
    gmsh::model::getBoundary({{3, v}}, bnd, false, false, false);

    int outerSurf = -1;
    double rmax = -std::numeric_limits<double>::infinity();
    for (const auto& p : bnd) {
      if (p.first != 2) continue;
      double r = MeanRadiusOfSurface(p.second);
      if (r > rmax) {
        rmax = r;
        outerSurf = p.second;
      }
    }
    if (outerSurf != -1) layers.emplace_back(v, outerSurf, rmax);
  }

  std::sort(layers.begin(), layers.end(), [](const auto& a, const auto& b) {
    return std::get<2>(a) < std::get<2>(b);
  });

  for (std::size_t i = 0; i < layers.size(); ++i) {
    const int physId = static_cast<int>(i) + 1;
    const int vTag = std::get<0>(layers[i]);
    const int sTag = std::get<1>(layers[i]);

    gmsh::model::addPhysicalGroup(3, {vTag}, physId);
    gmsh::model::setPhysicalName(3, physId, volPrefix + std::to_string(physId));

    gmsh::model::addPhysicalGroup(2, {sTag}, physId);
    gmsh::model::setPhysicalName(2, physId,
                                 surfPrefix + std::to_string(physId));
  }
}

double MeanRadiusOfSurface(int surfTag) {
  std::vector<std::size_t> nodeTags;
  std::vector<double> xyz;
  std::vector<double> param;
  gmsh::model::mesh::getNodes(nodeTags, xyz, param, 2, surfTag, true, false);
  if (xyz.empty()) return 0.0;
  double s = 0;
  size_t n = xyz.size() / 3;
  for (size_t i = 0; i < n; ++i) {
    double x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    s += std::sqrt(x * x + y * y + z * z);
  }

  return s / double(n);
}

LonLatField::LonLatField(std::vector<double> lons, std::vector<double> lats)
    : _lons(std::move(lons)),
      _lats(std::move(lats)),
      _nlon(static_cast<int>(_lons.size())),
      _nlat(static_cast<int>(_lats.size())) {}

size_t LonLatField::Idx(int i, int j) const {
  return static_cast<size_t>(j) * static_cast<size_t>(_nlon) +
         static_cast<size_t>(i);
}

double LonLatField::NorthPole(const std::vector<double>& field) const {
  const int j0 = _nlat - 2, j1 = _nlat - 1;
  const double y0 = _lats[j0], y1 = _lats[j1];
  const double dy = y1 - y0;
  const double t = (90.0 - y1) / dy + 1.0;
  double sum = 0.0;
  for (int i = 0; i < _nlon; ++i) {
    const double v0 = field[Idx(i, j0)];
    const double v1 = field[Idx(i, j1)];
    sum += v0 * (1.0 - t) + v1 * t;
  }
  return sum / _nlon;
}

double LonLatField::SouthPole(const std::vector<double>& field) const {
  const int j0 = 0, j1 = 1;
  const double y0 = _lats[j0], y1 = _lats[j1];
  const double dy = y1 - y0;
  const double t = (-90.0 - y0) / dy;
  double sum = 0.0;
  for (int i = 0; i < _nlon; ++i) {
    const double v0 = field[Idx(i, j0)];
    const double v1 = field[Idx(i, j1)];
    sum += v0 * (1.0 - t) + v1 * t;
  }
  return sum / _nlon;
}

double LonLatField::Bilerp(const std::vector<double>& field, double lon,
                           double lat) const {
  if (_nlon <= 1 || _nlat <= 1)
    throw std::runtime_error("LonLatField::bilerp requires nlon>1 and nlat>1");

  if (lat > 90.0) lat = 90.0;
  if (lat < -90.0) lat = -90.0;

  {
    double x = std::fmod(lon + 180.0, 360.0);
    if (x < 0.0) x += 360.0;
    lon = x - 180.0;
  }

  const double lonMin = _lons.front();
  const double lonMax = _lons.back();

  int i0, i1;
  double a;

  if (lon >= lonMin && lon <= lonMax) {
    int i_hi =
        int(std::lower_bound(_lons.begin(), _lons.end(), lon) - _lons.begin());
    if (i_hi == 0) {
      i0 = 0;
      i1 = 1;
    } else if (i_hi >= _nlon) {
      i0 = _nlon - 2;
      i1 = _nlon - 1;
    } else {
      i0 = i_hi - 1;
      i1 = i_hi;
    }
    const double x0 = _lons[i0], x1 = _lons[i1];
    a = (x1 != x0) ? (lon - x0) / (x1 - x0) : 0.0;
  } else {
    i0 = _nlon - 1;
    i1 = 0;
    const double seamWidth = (lonMin + 360.0) - lonMax;
    if (lon > lonMax)
      a = (lon - lonMax) / seamWidth;
    else
      a = ((lon + 360.0) - lonMax) / seamWidth;
  }

  if (lat > _lats.back()) {
    const int jt = _nlat - 1;
    const double y0 = _lats[jt];
    const double vTop = (1.0 - a) * field[Idx(i0, jt)] + a * field[Idx(i1, jt)];
    const double den = (90.0 - y0);
    const double t = (lat - y0) / den;
    return (1.0 - t) * vTop + t * NorthPole(field);
  }

  if (lat < _lats.front()) {
    const int jb = 0;
    const double y1 = _lats[jb];
    const double vBottom =
        (1.0 - a) * field[Idx(i0, jb)] + a * field[Idx(i1, jb)];
    const double den = (y1 - (-90.0));
    const double t = (lat - (-90.0)) / den;
    return (1.0 - t) * SouthPole(field) + t * vBottom;
  }

  int j_hi =
      int(std::lower_bound(_lats.begin(), _lats.end(), lat) - _lats.begin());
  int j0, j1;
  if (j_hi == 0) {
    j0 = 0;
    j1 = 1;
  } else if (j_hi >= _nlat) {
    j0 = _nlat - 2;
    j1 = _nlat - 1;
  } else {
    j0 = j_hi - 1;
    j1 = j_hi;
  }

  const double y0 = _lats[j0], y1 = _lats[j1];
  const double b = (y1 != y0) ? (lat - y0) / (y1 - y0) : 0.0;

  const double f00 = field[Idx(i0, j0)];
  const double f10 = field[Idx(i1, j0)];
  const double f01 = field[Idx(i0, j1)];
  const double f11 = field[Idx(i1, j1)];

  const double w00 = (1.0 - a) * (1.0 - b);
  const double w10 = a * (1.0 - b);
  const double w01 = (1.0 - a) * b;
  const double w11 = a * b;

  return f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11;
}

PREMModel::PREMModel(const std::string& fileName, double Rref,
                     double buffer_ratio, int ignored_layers)
    : _Rref(Rref),
      _buffer_ratio(buffer_ratio),
      _ignored_layers(ignored_layers) {
  if (_buffer_ratio < 0.0)
    throw std::invalid_argument("buffer_depth must be >= 0");

  std::ifstream file(fileName);
  if (!file) throw std::runtime_error("Unable to open PREM file: " + fileName);

  std::string line;
  bool dataStarted = false;
  double prevR = std::numeric_limits<double>::quiet_NaN();

  auto try_parse_line = [&](const std::string& ln) {
    std::istringstream iss(ln);
    double r, density, pWave, sWave, bulkM, shearM;
    if (!(iss >> r >> density >> pWave >> sWave >> bulkM >> shearM)) {
      throw std::runtime_error("PREM bad line: " + ln);
    }

    if (!std::isnan(prevR) && std::abs(r - prevR) / prevR < 1e-6) {
      radii.push_back(r);
    }
    prevR = r;
  };

  while (std::getline(file, line)) {
    if (!dataStarted) {
      std::istringstream probe(line);
      std::string firstTok;
      if (!(probe >> firstTok)) continue;
      if (firstTok == "0.") {
        dataStarted = true;
        try_parse_line(line);
      }
    } else {
      try_parse_line(line);
    }
  }

  if (_ignored_layers < 0 ||
      static_cast<std::size_t>(_ignored_layers) > radii.size()) {
    throw std::out_of_range("ignored_layers out of range");
  }

  const std::size_t keepN =
      radii.size() - static_cast<std::size_t>(_ignored_layers);
  radii_nd.reserve(keepN + 1);

  for (std::size_t i = 0; i < keepN; ++i) radii_nd.push_back(radii[i] / _Rref);

  radii_nd.push_back(1 + _buffer_ratio);
}

std::vector<double>& PREMModel::GetRadiiND() { return radii_nd; }
std::vector<double>& PREMModel::GetRadii() { return radii; }

Topography::Topography(const std::string& xyzFile, double Rref) : _Rref(Rref) {
  std::vector<double> L, B, V;
  if (!LoadXYZ(xyzFile, L, B, V))
    throw std::runtime_error("Topography: cannot read " + xyzFile);

  for (double &lon : L) {
    lon = std::fmod(lon + 180.0, 360.0);
    if (lon < 0) lon += 360.0;
    lon -= 180.0;
  }

  for (double& v : V) v /= _Rref;
  BuildGrid(L, B, V);
}

Topography& Topography::operator+=(const Topography& other) {
  for (int j = 0; j < _grid.NLat(); ++j) {
    const double lat = LatAt(j);
    for (int i = 0; i < _grid.NLon(); ++i) {
      const double lon = LonAt(i);
      const double va = _data[_grid.Idx(i, j)];
      const double vb = other.Interp(lon, lat);
      if (std::isfinite(va) && std::isfinite(vb))
        _data[_grid.Idx(i, j)] = va + vb;
      else
        throw std::runtime_error("Infinite value in Topography::operator+=");
    }
  }
  return *this;
}

Topography operator+(const Topography& A, const Topography& B) {
  std::vector<double> V(
      static_cast<size_t>(A._grid.NLon()) * static_cast<size_t>(A._grid.NLat()),
      std::numeric_limits<double>::quiet_NaN());
  for (int j = 0; j < A._grid.NLat(); ++j) {
    const double lat = A.LatAt(j);
    for (int i = 0; i < A._grid.NLon(); ++i) {
      const double lon = A.LonAt(i);
      const double va = A._data[A._grid.Idx(i, j)];
      const double vb = B.Interp(lon, lat);
      if (std::isfinite(va) && std::isfinite(vb))
        V[A._grid.Idx(i, j)] = va + vb;
      else
        throw std::runtime_error("Infinite value in Topography::operator+");
    }
  }
  return Topography(A._grid.Lons(), A._grid.Lats(), A._Rref, std::move(V));
}

double Topography::Interp(double lon, double lat) const {
  return _grid.Bilerp(_data, lon, lat);
}

int Topography::NLon() const { return _grid.NLon(); }
int Topography::NLat() const { return _grid.NLat(); }
const std::vector<double>& Topography::Lons() const { return _grid.Lons(); }
const std::vector<double>& Topography::Lats() const { return _grid.Lats(); }
const std::vector<double>& Topography::Data() const { return _data; }
double Topography::LonAt(int i) const { return _grid.LonAt(i); }
double Topography::LatAt(int j) const { return _grid.LatAt(j); }

double Topography::Mean() const {
  double sum = 0.0;
  size_t n = 0;
  for (double v : _data)
    if (std::isfinite(v)) {
      sum += v;
      ++n;
    }
  return n ? (sum / double(n)) : 0.0;
}

Topography::Topography(std::vector<double> lons, std::vector<double> lats,
                       double Rref, std::vector<double> data)
    : _grid(LonLatField(std::move(lons), std::move(lats))),
      _Rref(Rref),
      _data(std::move(data)) {}

bool Topography::LoadXYZ(const std::string& file, std::vector<double>& L,
                         std::vector<double>& B, std::vector<double>& V) {
  std::ifstream in(file);
  if (!in) return false;
  L.clear();
  B.clear();
  V.clear();
  double a, b, c;
  while (in >> a >> b >> c) {
    L.push_back(a);
    B.push_back(b);
    V.push_back(c);
  }
  return !L.empty();
}

void Topography::BuildGrid(const std::vector<double>& L,
                           const std::vector<double>& B,
                           const std::vector<double>& V) {
  if (L.size() != B.size() || L.size() != V.size())
    throw std::runtime_error("Topography::buildGrid: xyz size mismatch");

  std::vector<double> lons = L, lats = B;
  std::sort(lons.begin(), lons.end());
  lons.erase(std::unique(lons.begin(), lons.end()), lons.end());
  std::sort(lats.begin(), lats.end());
  lats.erase(std::unique(lats.begin(), lats.end()), lats.end());

  _grid = LonLatField(std::move(lons), std::move(lats));
  _data.assign(
      static_cast<size_t>(_grid.NLon()) * static_cast<size_t>(_grid.NLat()),
      std::numeric_limits<double>::quiet_NaN());

  const double tol = 1e-8;

  for (size_t k = 0; k < L.size(); ++k) {
    const double lon = L[k];
    const double lat = B[k];

    if (lon < _grid.Lons().front() - tol || lon > _grid.Lons().back() + tol ||
        lat < _grid.Lats().front() - tol || lat > _grid.Lats().back() + tol) {
      std::ostringstream oss;
      oss << "Topography::buildGrid: point (" << lon << ", " << lat
          << ") out of grid range";
      throw std::runtime_error(oss.str());
    }

    auto itx = std::lower_bound(_grid.Lons().begin(), _grid.Lons().end(), lon);
    auto ity = std::lower_bound(_grid.Lats().begin(), _grid.Lats().end(), lat);

    if (itx == _grid.Lons().end()) --itx;
    if (ity == _grid.Lats().end()) --ity;

    if (std::fabs(*itx - lon) > tol || std::fabs(*ity - lat) > tol) {
      std::ostringstream oss;
      oss << "Topography::buildGrid: (" << lon << "," << lat
          << ") not aligned to grid centers";
      throw std::runtime_error(oss.str());
    }

    const int i = static_cast<int>(itx - _grid.Lons().begin());
    const int j = static_cast<int>(ity - _grid.Lats().begin());
    _data[_grid.Idx(i, j)] = V[k];
  }
}

FieldRadialSurface::FieldRadialSurface(const LonLatField& grid,
                                       const std::vector<double>& r_field)
    : _grid(grid), _r_field(r_field) {}

double FieldRadialSurface::RadiusAt(double lon, double lat) const {
  return _grid.Bilerp(_r_field, lon, lat);
}

SpheroidalRadialSurface::SpheroidalRadialSurface(double r) : _r(r) {}
double SpheroidalRadialSurface::RadiusAt(double, double) const { return _r; }

EllipsoidalRadialSurface::EllipsoidalRadialSurface(double a, double b, double c)
    : _a(a), _b(b), _c(c) {}
double EllipsoidalRadialSurface::RadiusAt(double lon, double lat) const {
  const double L = Deg2Rad(lon), B = Deg2Rad(lat);
  const double nx = std::cos(B) * std::cos(L);
  const double ny = std::cos(B) * std::sin(L);
  const double nz = std::sin(B);
  const double denom =
      (nx * nx) / (_a * _a) + (ny * ny) / (_b * _b) + (nz * nz) / (_c * _c);
  return (denom > 0.0) ? 1.0 / std::sqrt(denom) : 0.0;
}

RadialMapping::RadialMapping(const std::vector<const Topography*>& topo,
                             double topo_exag)
    : _topo(topo), _topo_exag(topo_exag) {}

double RadialMapping::InterpTopo(std::size_t i, double lon, double lat) const {
  return _topo[i]->Interp(lon, lat) * _topo_exag;
}

CubicBandLinearDecay::CubicBandLinearDecay(
    const std::vector<const Topography*>& topo,
    const std::vector<const RadialSurface*>& base, double decay,
    double topo_exag, std::size_t iInner, std::size_t iOuter)
    : RadialMapping(topo, topo_exag),
      _base(base),
      _decay(decay),
      _iInner(iInner),
      _iOuter(iOuter) {}

double CubicBandLinearDecay::Displacement(double r, double lon,
                                          double lat) const {
  const double rin = _base[_iInner]->RadiusAt(lon, lat);
  const double rout = _base[_iOuter]->RadiusAt(lon, lat);
  const double dInner = InterpTopo(_iInner, lon, lat);
  const double dOuter = InterpTopo(_iOuter, lon, lat);

  const double r_in_lo = rin - _decay;
  const double r_mid_lo = rin;
  const double r_mid_hi = rout;
  const double r_out_hi = rout + _decay;

  if (r <= 0.0 || r < r_in_lo || r > r_out_hi) return 0.0;

  if (r <= r_mid_lo) {
    double t = (rin - r) / _decay;
    t = std::clamp(t, 0.0, 1.0);
    return dInner * (1.0 - t);
  } else if (r < r_mid_hi) {
    double t = (r - rin) / (rout - rin);
    t = std::clamp(t, 0.0, 1.0);
    const double w = 1.0 - t * t * (3.0 - 2.0 * t);
    return w * dInner + (1.0 - w) * dOuter;
  } else {
    double t = (r - rout) / _decay;
    t = std::clamp(t, 0.0, 1.0);
    return dOuter * (1.0 - t);
  }
}

void PerturbAllNodes(const RadialMapping& mapping) {
  std::vector<std::size_t> tags;
  std::vector<double> xyz, param;
  gmsh::model::mesh::getNodes(tags, xyz, param, -1, -1, true, false);

  for (std::size_t i = 0; i < tags.size(); ++i) {
    double& x = xyz[3 * i + 0];
    double& y = xyz[3 * i + 1];
    double& z = xyz[3 * i + 2];

    const double r = std::sqrt(x * x + y * y + z * z);
    if (r == 0.0) continue;

    const double lon = Rad2Deg(std::atan2(y, x));
    const double lat = Rad2Deg(std::asin(z / r));

    const double disp = mapping.Displacement(r, lon, lat);
    if (!std::isfinite(disp)) {
      std::ostringstream oss;
      oss << "Non-finite displacement at node " << tags[i] << " (lon=" << lon
          << ", lat=" << lat << ", r=" << r << ")";
      throw std::runtime_error(oss.str());
    }
    if (r + disp <= 0.0) {
      std::ostringstream oss;
      oss << "Negative or zero resulting radius at node " << tags[i]
          << " (lon=" << lon << ", lat=" << lat << ", r=" << r
          << ", disp=" << disp << ")";
      throw std::runtime_error(oss.str());
    }

    const double s = (r + disp) / r;
    x *= s;
    y *= s;
    z *= s;
    gmsh::model::mesh::setNode(tags[i], {x, y, z}, {});
  }
}


