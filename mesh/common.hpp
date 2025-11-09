#pragma once

#include <gmsh.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
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

// Helper function to tag all volumes and surfaces with the 'average radius' in
// the ascending order. First is the list of volume tags, second and third are
// the prefixes of volume and surface tags, respectively.
void TagLayersByRadius(const std::vector<int>& volTags,
                       const std::string& volPrefix = "volume_",
                       const std::string& surfPrefix = "surface_");

// Helper function to calculate the mean radius of a surface as an arithmetic
// average over all nodes. Returns a double: first is the tag of the surface to
// be measured
double MeanRadiusOfSurface(int surfTag);

/**
 * @brief A global timer for convenience.
 */
struct Timer {
  using clk = std::chrono::steady_clock;
  /** @brief Initial time when the timer is constructed. */
  clk::time_point t0 = clk::now();
  /** @brief Last time stop. */
  clk::time_point last = t0;
  /**
   * @brief Output the time since the previous mark and total time since
   * construction.
   *
   * Prints in the format:
   * `[TIMER] +<dt> s (<tot> s total): <msg>`
   *
   * @param msg A message describing the section that just completed.
   *
   * @note Uses a monotonic clock (steady_clock). Units are seconds.
   * @post Updates the internal @ref last timestamp to 'now'.
   */
  void Mark(const std::string& msg) {
    auto now = clk::now();
    double dt = std::chrono::duration<double>(now - last).count();
    double tot = std::chrono::duration<double>(now - t0).count();
    last = now;
    std::cout << std::fixed << std::setprecision(3) << "[TIMER] +" << dt
              << " s (" << tot << " s total): " << msg << "\n";
  }
};

/**
 * @brief Parse doubles separated by `-` (e.g. mesh sizes).
 *
 * Splits on '-' characters, trims whitespace in each token, and converts
 * non-empty tokens to double.
 *
 * @param s A string in CLI (e.g., "1.0-0.5-0.25").
 * @return A std::vector<double> of parsed double values.
 */
inline std::vector<double> ParseDoubles(const std::string& s) {
  std::vector<double> v;
  std::istringstream iss(s);
  std::string tok;
  while (std::getline(iss, tok, '-')) {
    tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
    if (!tok.empty()) v.push_back(static_cast<double>(std::stod(tok)));
  }
  return v;
}

/**
 * @brief An inline function transforming degree to radian.
 *
 * @param d Degree value.
 * @return r Radian value.
 *
 * @note Uses std::numbers::pi. Input/Output are double.
 */
inline double Deg2Rad(double d) { return d * std::numbers::pi / 180.0; }

/**
 * @brief An inline function transforming radian to degree.
 *
 * @param r Radian value.
 * @return Degree value.
 *
 * @note Uses std::numbers::pi. Input/Output are double.
 */
inline double Rad2Deg(double r) { return r * 180.0 / std::numbers::pi; }

/**
 * @brief A class representing the lon-lat grid.
 *
 * Stores 1D arrays of longitudes and latitudes and provides index mapping
 * and bilinear interpolation support for fields sampled on the tensor grid.
 */
class LonLatField {
 public:
  /** @brief Default-constructed empty grid. */
  LonLatField() = default;

  /**
   * @brief Constructs a LonLatFied with lists of lon- and lat- coordinates.
   *
   * @param lons Longitudes in degrees.
   * @param lats Latitudes in degrees.
   */
  LonLatField(std::vector<double> lons, std::vector<double> lats);

  /** @brief Number of longitude nodes. */
  int NLon() const { return _nlon; }
  /** @brief Number of latitude nodes. */
  int NLat() const { return _nlat; }

  /** @brief Read-only access to longitudes. */
  const std::vector<double>& Lons() const { return _lons; }
  /** @brief Read-only access to latitudes. */
  const std::vector<double>& Lats() const { return _lats; }

  /** @brief Longitude at index @p i. */
  double LonAt(int i) const { return _lons[i]; }
  /** @brief Latitude at index @p j. */
  double LatAt(int j) const { return _lats[j]; }

  /**
   * @brief Convert 2D indices (i,j) to a flat array index.
   *
   * @param i Longitude index in [0, NLon()).
   * @param j Latitude index in [0, NLat()).
   * @return Flattened index for row major layout.
   */
  size_t Idx(int i, int j) const;

  /**
   * @brief Evaluate a field value at the North Pole from field values at the
   * highest latitude.
   *
   * @param field Flattened field values of size NLon()*NLat().
   * @return A interpolated value at lat = +90°.
   */
  double NorthPole(const std::vector<double>& field) const;

  /**
   * @brief Evaluate a field value at the South Pole from field values at the
   * lowest latitude.
   *
   * @param field Flattened field values of size NLon()*NLat().
   * @return A interpolated value at lat = -90°.
   */
  double SouthPole(const std::vector<double>& field) const;

  /**
   * @brief Bilinear interpolation on the lon-lat grid.
   *
   * @param field Flattened field values of size NLon()*NLat().
   * @param lon Query longitude (degrees).
   * @param lat Query latitude (degrees).
   * @return Interpolated value at (lon, lat).
   */
  double Bilerp(const std::vector<double>& field, double lon, double lat) const;

 private:
  std::vector<double> _lons, _lats;  ///< Coordinate arrays (degrees).
  int _nlon = 0, _nlat = 0;          ///< Grid dimensions.
};

/**
 * @brief PREM (Preliminary Reference Earth Model) radial profiles and
 * properties.
 *
 * Loads a PREM-like text file, skipping lines until the first line starting
 * with '0.'. Stores boundary radii (dimensional and non-dimensional) and
 * optional property lists.
 */
class PREMModel {
 public:
  /**
   * @brief Construct and load PREM data.
   *
   * @param fileName Path to a PREM-formatted text file.
   * @param Rref Reference radius used for non-dimensionalization.
   * @param buffer_ratio Radio of the buffer layer depth over the reference
   * radius.
   * @param ignored_layers Number of outermost layers to ignore (default = 0).
   */
  PREMModel(const std::string& fileName, double Rref, double buffer_ratio,
            int ignored_layers = 0);

  /**
   * @brief Access non-dimensional radii.
   * @return Reference to radii divided by @ref _Rref.
   */
  std::vector<double>& GetRadiiND();

  /**
   * @brief Access dimensional radii.
   * @return Reference to radii in the same units as @ref _Rref.
   */
  std::vector<double>& GetRadii();

  ~PREMModel() = default;

 private:
  double _Rref;          ///< Reference radius for non-dimensionalization.
  double _buffer_ratio;  ///< Radio of the buffer layer depth over the reference
                         ///< radius.
  int _ignored_layers;   ///< Number of outermost layers ignored on import.

  std::vector<double> radii;     ///< Dimensional layer boundary radii.
  std::vector<double> radii_nd;  ///< Non-dimensional radii (r / @ref _Rref).
  std::vector<double> density_list, pWave_list, sWave_list, bulkM_list,
      shearM_list;  ///< Optional property tables.
};

/**
 * @brief Gridded scalar surface (e.g., topography) sampled on a lon-lat grid.
 *
 * Supports loading from XYZ (lon lat value) files, interpolation, arithmetic,
 * and basic stats.
 */
class Topography {
 public:
  Topography() = default;
  Topography(const Topography&) = default;
  Topography(Topography&&) noexcept = default;
  Topography& operator=(const Topography&) = default;
  Topography& operator=(Topography&&) noexcept = default;

  /**
   * @brief Construct from a XYZ file.
   *
   * @param xyzFile Path to a file with lines: "lon lat value" (degrees for
   * lon/lat).
   * @param Rref Length scale for non-dimensionalisation (default 1.0).
   */
  Topography(const std::string& xyzFile, double Rref = 1.0);

  /**
   * @brief In-place pointwise addition.
   * @param other Another topography.
   * @return Reference to *this.
   *
   * @pre Grids can be on different coordinates but only the coordinates of
   * *this are inherited.
   */
  Topography& operator+=(const Topography& other);

  /**
   * @brief Pointwise addition of two topographies.
   * @param A Left operand.
   * @param B Right operand.
   * @return A new Topography equal to A + B.
   *
   * @pre @pre Grids can be on different coordinates but only the coordinates of
   * the left are inherited.
   */
  friend Topography operator+(const Topography& A, const Topography& B);

  /**
   * @brief Interpolate the surface value at (lon, lat).
   * @param lon Longitude in degrees.
   * @param lat Latitude in degrees.
   * @return Interpolated value.
   */
  double Interp(double lon, double lat) const;

  /** @brief Number of longitude samples. */
  int NLon() const;
  /** @brief Number of latitude samples. */
  int NLat() const;
  /** @brief Read-only longitudes. */
  const std::vector<double>& Lons() const;
  /** @brief Read-only latitudes. */
  const std::vector<double>& Lats() const;
  /** @brief Read-only flattened data array of size NLon()*NLat(). */
  const std::vector<double>& Data() const;

  /** @brief Longitude at index i. */
  double LonAt(int i) const;
  /** @brief Latitude at index j. */
  double LatAt(int j) const;

  /**
   * @brief Arithmetic mean of all grid values.
   * @return The mean value over @ref _data.
   */
  double Mean() const;

 private:
  LonLatField _grid;          ///< Underlying lon-lat grid.
  double _Rref = 1.0;         ///< Length scale for non-dimensionalisation.
  std::vector<double> _data;  ///< Flattened data (size NLon()*NLat()).

  /**
   * @brief Construct directly from vectors.
   *
   * @param lons Longitudes (degrees).
   * @param lats Latitudes (degrees).
   * @param Rref Length scale.
   * @param data Flattened data of size lons.size() * lats.size().
   */
  Topography(std::vector<double> lons, std::vector<double> lats, double Rref,
             std::vector<double> data);

  /**
   * @brief Load XYZ into separate arrays.
   *
   * @param file Path to file.
   * @param L Output longitudes (degrees).
   * @param B Output latitudes (degrees).
   * @param V Output field values.
   * @return true on success, false on failure.
   */
  static bool LoadXYZ(const std::string& file, std::vector<double>& L,
                      std::vector<double>& B, std::vector<double>& V);

  /**
   * @brief Build a structured grid and data from scattered XYZ vectors.
   *
   * @param L Longitudes (degrees).
   * @param B Latitudes (degrees).
   * @param V Values aligned with (L,B).
   */
  void BuildGrid(const std::vector<double>& L, const std::vector<double>& B,
                 const std::vector<double>& V);
};

/**
 * @brief Abstract interface for a radial surface r = R(lon, lat).
 */
class RadialSurface {
 public:
  virtual ~RadialSurface() = default;
  /**
   * @brief Evaluate the radius at a given direction.
   *
   * @param lon Longitude (degrees).
   * @param lat Latitude (degrees).
   * @return Radius value.
   */
  virtual double RadiusAt(double lon, double lat) const = 0;
  // virtual double MeanRadius() const = 0;
};

/**
 * @brief Radial surface defined by a sampled field on a lon-lat grid.
 */
class FieldRadialSurface final : public RadialSurface {
 public:
  /**
   * @brief Construct from a grid and corresponding radius field.
   *
   * @param grid Underlying lon-lat grid.
   * @param r_field Flattened radius field of size grid.NLon() * grid.NLat().
   */
  FieldRadialSurface(const LonLatField& grid,
                     const std::vector<double>& r_field);
  /** @copydoc RadialSurface::RadiusAt */
  double RadiusAt(double lon, double lat) const override;

 private:
  const LonLatField& _grid;  ///< Reference to the sampling grid.
  const std::vector<double>&
      _r_field;  ///< Reference to the sampled radial field.
};

/**
 * @brief Spherical (constant-radius) surface r = const.
 */
class SpheroidalRadialSurface final : public RadialSurface {
 public:
  /**
   * @brief Construct with constant radius.
   * @param r Sphere radius.
   */
  explicit SpheroidalRadialSurface(double r);
  /** @copydoc RadialSurface::RadiusAt */
  double RadiusAt(double, double) const override;

 private:
  double _r;  ///< Constant radius.
};

/**
 * @brief Triaxial ellipsoidal surface.
 *
 * Parameterized by semi-axes (a,b,c); returns the distance to the ellipsoid in
 * the (lon,lat) direction.
 */
class EllipsoidalRadialSurface final : public RadialSurface {
 public:
  /**
   * @brief Construct an ellipsoidal surface.
   * @param a Semi-axis along x.
   * @param b Semi-axis along y.
   * @param c Semi-axis along z.
   */
  EllipsoidalRadialSurface(double a, double b, double c);
  /** @copydoc RadialSurface::RadiusAt */
  double RadiusAt(double lon, double lat) const override;

 private:
  double _a, _b, _c;  ///< Ellipsoid semi-axes.
};

// RadialMapping
/**
 * @brief Base class for radial displacement mappings driven by a set of
 * topographies.
 */
class RadialMapping {
 public:
  /**
   * @brief Construct with a collection of topographies.
   *
   * @param topo Vector of pointers to topography fields (ownership not taken).
   * @param topo_exag Multiplicative exaggeration applied to topography when
   * used (default 1.0).
   */
  RadialMapping(const std::vector<const Topography*>& topo,
                double topo_exag = 1.0);
  virtual ~RadialMapping() = default;

  /**
   * @brief Radial displacement at (r, lon, lat).
   *
   * @param r   Query radius (same units as topography reference).
   * @param lon Longitude (degrees).
   * @param lat Latitude (degrees).
   * @return Radial displacement (positive outward).
   */
  virtual double Displacement(double r, double lon, double lat) const = 0;

 protected:
  /**
   * @brief Interpolate topography field @p i at (lon, lat) with exaggeration
   * applied.
   *
   * @param i Index into @ref _topo.
   * @param lon Longitude (degrees).
   * @param lat Latitude (degrees).
   * @return Interpolated (and possibly scaled) topography value.
   *
   * @pre 0 <= i < _topo.size()
   */
  double InterpTopo(std::size_t i, double lon, double lat) const;

  const std::vector<const Topography*>&
      _topo;          ///< External references to Topographies.
  double _topo_exag;  ///< Exaggeration factor applied to topography.
};

/**
 * @brief Cubic band mapping with linear decay outside a [inner, outer] shell.
 *
 * Uses topographies referenced to two base radial surfaces, applying a cubic
 * profile within the radial band and a linear decay within depth @ref _decay
 * outside.
 */
class CubicBandLinearDecay final : public RadialMapping {
 public:
  /**
   * @brief Construct the mapping.
   *
   * @param topo      Topography fields used to drive the mapping.
   * @param base      Base radial surfaces.
   * @param decay     Layer depth for linear decay.
   * @param topo_exag Topography exaggeration factor (default 1.0).
   * @param iInner    Index into @p base for the inner surface (default 0).
   * @param iOuter    Index into @p base for the outer surface (default 1).
   */
  CubicBandLinearDecay(const std::vector<const Topography*>& topo,
                       const std::vector<const RadialSurface*>& base,
                       double decay, double topo_exag = 1.0,
                       std::size_t iInner = 0, std::size_t iOuter = 1);

  /**
   * @brief Radial displacement with cubic behavior in-band and linear decay
   * out-of-band.
   *
   * @param r   Query radius.
   * @param lon Longitude (degrees).
   * @param lat Latitude (degrees).
   * @return Displacement (positive outward).
   */
  double Displacement(double r, double lon, double lat) const override;

 private:
  const std::vector<const RadialSurface*>&
      _base;                ///< Base radial surfaces (no ownership).
  double _decay = 0.0;      ///< Decay layer depth.
  std::size_t _iInner = 0;  ///< Index of inner base surface.
  std::size_t _iOuter = 1;  ///< Index of outer base surface.
};

// Helper function to perturb radially all nodes in a mesh with the scheme
// RadialMapping. mapping: a specific RadialMapping class
void PerturbAllNodes(const RadialMapping& mapping);
