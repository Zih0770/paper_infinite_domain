// src/common.hpp
#pragma once

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

#include <iomanip>
#include <filesystem>

#include <chrono>

using namespace mfem;

class Nondimensionalisation {
private:
    real_t L;   // Length scale [m]
    real_t T;   // Time scale [s]
    real_t RHO; // Density scale [kg/m^3]

public:
    Nondimensionalisation(real_t length_scale, real_t time_scale, real_t density_scale)
        : L(length_scale), T(time_scale), RHO(density_scale) {}

    // Accessors
    real_t Length() const { return L; }
    real_t Time() const { return T; }
    real_t Density() const { return RHO; }

    // Derived scales
    real_t Velocity() const { return L / T; }
    real_t Acceleration() const { return L / (T*T); }
    real_t Pressure() const { return RHO * L*L / (T*T); } // [Pa]
    real_t Gravity() const { return L / (T*T); }
    real_t Potential() const { return L*L / (T*T); }

    // Scaling functions for scalars
    real_t ScaleLength(real_t x) const { return x / L; }
    real_t UnscaleLength(real_t x_nd) const { return x_nd * L; }

    real_t ScaleDensity(real_t rho) const { return rho / RHO; }
    real_t UnscaleDensity(real_t rho_nd) const { return rho_nd * RHO; }

    real_t ScaleGravityPotential(real_t phi) const { return phi / Potential(); }
    real_t UnscaleGravityPotential(real_t phi_nd) const { return phi_nd * Potential(); }

    real_t ScaleStress(real_t sigma) const { return sigma / Pressure(); }
    real_t UnscaleStress(real_t sigma_nd) const { return sigma_nd * Pressure(); }

    // Scaling for GridFunction fields
    void UnscaleGravityPotential(GridFunction &phi_gf) const { phi_gf *= Potential(); }
    void UnscaleDisplacement(GridFunction &u_gf) const { u_gf *= L; }
    void UnscaleStress(GridFunction &sigma_gf) const { sigma_gf *= Pressure(); }

    // Create a scaled density coefficient from a dimensional one
    Coefficient *MakeScaledDensityCoefficient(Coefficient &rho_coeff) const {
        return new ProductCoefficient(1.0 / RHO, rho_coeff);
    }

    void Print() const {
        std::cout << "Scaling parameters:\n";
        std::cout << "  Length scale: " << L << " m\n";
        std::cout << "  Time scale: " << T << " s\n";
        std::cout << "  Density scale: " << RHO << " kg/m^3\n";
        std::cout << "  Gravity potential scale: " << Potential() << " m^2/s^2\n";
    }
};

static double EvalCoefficientAtPointGlobal(Coefficient &coef,
                                           ParMesh &pmesh,
                                           const Vector &x)
{
  double loc_val = 0.0; int loc_has = 0;
  for (int e = 0; e < pmesh.GetNE(); ++e) {
    ElementTransformation *T = pmesh.GetElementTransformation(e);
    InverseElementTransformation inv(T);
    IntegrationPoint ip;
    if (inv.Transform(x, ip) == InverseElementTransformation::Inside) {
      T->SetIntPoint(&ip);
      loc_val = coef.Eval(*T, ip);
      loc_has = 1;
      break;
    }
  }
  double glob_val = 0.0; int glob_has = 0;
  MPI_Allreduce(&loc_val, &glob_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_has, &glob_has, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (glob_has > 0) { glob_val /= glob_has; }
  return glob_val;
}

static double SampledValueAtGlobal(const ParGridFunction &gf, const Vector &c0,
                                   double r, int n)
{
  double loc_sum = 0.0; int loc_cnt = 0;

  auto try_add = [&](const Vector &x){
    auto *pmesh = gf.ParFESpace()->GetParMesh();
    for (int e = 0; e < pmesh->GetNE(); ++e) {
      ElementTransformation *T = pmesh->GetElementTransformation(e);
      InverseElementTransformation inv(T);
      IntegrationPoint ip;
      if (inv.Transform(x, ip) == InverseElementTransformation::Inside) {
        T->SetIntPoint(&ip);
        loc_sum += gf.GetValue(*T, ip);
        loc_cnt += 1;
        return;
      }
    }
  };

  try_add(c0);

  const int dim = gf.ParFESpace()->GetParMesh()->Dimension();
  for (int k = 0; k < n; ++k) {
    const double th = 2.0 * std::numbers::pi * k / n;
    Vector x(c0.Size()); x = c0;
    if (dim == 2) { x[0] += r * cos(th); x[1] += r * sin(th); }
    else          { x[0] += r * cos(th); x[2] += r * sin(th); }
    try_add(x);
  }

  double glob_sum = 0.0; int glob_cnt = 0;
  MPI_Allreduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_cnt, &glob_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return (glob_cnt > 0) ? (glob_sum / glob_cnt) : 0.0;
}

static double ComputeRelL2_OnAttribute(const ParGridFunction &phi,
                                       const ParGridFunction &exact_gf,
                                       ParMesh &pmesh, int small_attr_idx)
{
  Array<int> attr_marker(pmesh.attributes.Max());
  attr_marker = 0;

  MFEM_ASSERT(small_attr_idx >= 0 && small_attr_idx < attr_marker.Size(),
              "small_attr index out of range.");
  attr_marker[small_attr_idx] = 1;

  GridFunctionCoefficient exact_coef(&exact_gf);
  ConstantCoefficient zero_c(0.0);

  const double num = phi.ComputeL2Error(exact_coef, /*irs*/nullptr, &attr_marker);
  const double den = exact_gf.ComputeL2Error(zero_c, /*irs*/nullptr, &attr_marker);

  return (den > 0.0) ? (num / den) : 0.0;
}

struct RelErrSignedCoefficient : public Coefficient {
  const ParGridFunction &phi, &exact; 
  double eps;
  RelErrSignedCoefficient(const ParGridFunction &ph, const ParGridFunction &ex, double e=1e-14)
    : phi(ph), exact(ex), eps(e) {}
  double Eval(ElementTransformation &T, const IntegrationPoint &ip) override {
    const double v  = phi.GetValue(T, ip);
    const double ve = exact.GetValue(T, ip);
    return (v - ve) / std::max(std::abs(ve), eps);
  }
};

struct LogAbsCoefficient : public Coefficient {
  const ParGridFunction &f; double eps;
  LogAbsCoefficient(const ParGridFunction &ff, double e=1e-14) : f(ff), eps(e) {}
  double Eval(ElementTransformation &T, const IntegrationPoint &ip) override {
    const double v = f.GetValue(T, ip);
    return std::log10(std::abs(v) + eps);
  }
};

static void SaveOnSmallDomainSubmesh(const std::string &tag, int order, int small_attr,
                                     ParMesh &pmesh,
                                     const ParGridFunction &phi_field,
                                     const ParGridFunction &exact_field,
                                     double relL2_small)
{
  const int world_rank = Mpi::WorldRank();

  Array<int> marker(pmesh.attributes.Max()); marker = 0;
  marker[small_attr] = 1;
  ParSubMesh psub = ParSubMesh::CreateFromDomain(pmesh, marker);

  H1_FECollection sfec(order, psub.Dimension());
  ParFiniteElementSpace sfes(&psub, &sfec);

  ParGridFunction phi_sm(&sfes), exact_sm(&sfes);
  phi_sm = 0.0; exact_sm = 0.0;
  psub.Transfer(phi_field,  phi_sm);
  psub.Transfer(exact_field, exact_sm);

  ParGridFunction resid_sm(&sfes);
  RelErrSignedCoefficient rcoef(phi_sm, exact_sm, 1e-14);
  resid_sm.ProjectCoefficient(rcoef);

  ParGridFunction resid_logabs_sm(&sfes);
  LogAbsCoefficient logabs_coef(resid_sm, 1e-14);
  resid_logabs_sm.ProjectCoefficient(logabs_coef);

  const int has_local = (psub.GetNE() > 0) ? 1 : 0;

  MPI_Comm subcomm = MPI_COMM_NULL;
  const int color = has_local ? 1 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subcomm);

  int subrank = -1, subsize = 0;
  if (has_local) {
    MPI_Comm_rank(subcomm, &subrank);
    MPI_Comm_size(subcomm, &subsize);
  }

  if (world_rank == 0) { std::filesystem::create_directories("results"); }

  if (has_local) {
    std::ostringstream mname, gphi, gex, gres, glog;
    mname << "results/" << tag << ".sm.mesh."     << std::setw(6) << std::setfill('0') << subrank;
    gphi  << "results/" << tag << ".sm.phi."      << std::setw(6) << std::setfill('0') << subrank;
    gex   << "results/" << tag << ".sm.exact."    << std::setw(6) << std::setfill('0') << subrank;
    gres  << "results/" << tag << ".sm.resid."    << std::setw(6) << std::setfill('0') << subrank;
    glog  << "results/" << tag << ".sm.residlog." << std::setw(6) << std::setfill('0') << subrank;

    { std::ofstream(mname.str()) << std::setprecision(8) << psub;
      std::ofstream f1(gphi.str());  f1.precision(8);  phi_sm.Save(f1);
      std::ofstream f2(gex.str());   f2.precision(8);  exact_sm.Save(f2);
      std::ofstream f3(gres.str());  f3.precision(8);  resid_sm.Save(f3);
      std::ofstream f4(glog.str());  f4.precision(8);  resid_logabs_sm.Save(f4); }

    if (subrank == 0) {
      std::ofstream info("results/" + tag + ".sm.info.txt");
      info << "glvis_np " << subsize << "\n"
           << "glvis_cmd glvis -np " << subsize
           << " -m results/" << tag << ".sm.mesh"
           << " -g results/" << tag << ".sm.resid\n";
      std::cout << "[GLVis] submesh writers (np) = " << subsize << "\n"
                << "        glvis -np " << subsize
                << " -m results/" << tag << ".sm.mesh"
                << " -g results/" << tag << ".sm.resid\n";
    }

    MPI_Comm_free(&subcomm);
  }

  if (Mpi::WorldRank() == 0) {
    std::ofstream rL2("results/" + tag + ".sm.relL2.txt");
    rL2 << std::setprecision(16)
        << "relative_L2_error_on_small_domain " << relL2_small << "\n";
  }
}


