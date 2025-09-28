// src/Poisson_big_domain.cpp
#include "mfem.hpp"
#include <mpi.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "uniform_sphere.hpp"
#include "common.hpp"

// Use your mesh utilities
#include "mfemElasticity/mesh.hpp"

using namespace mfem;
using namespace std;
using clk = std::chrono::steady_clock;

namespace {

// ---------------- helpers ----------------

static double EvalCoefficientAtPointGlobal(Coefficient &coef,
                                           ParMesh &pmesh,
                                           const Vector &x)
{
  double loc_val = 0.0;
  int    loc_has = 0;

  for (int e = 0; e < pmesh.GetNE(); ++e)
  {
    ElementTransformation *T = pmesh.GetElementTransformation(e);
    InverseElementTransformation inv(T);
    IntegrationPoint ip;
    if (inv.Transform(x, ip) == InverseElementTransformation::Inside)
    {
      T->SetIntPoint(&ip);
      loc_val = coef.Eval(*T, ip); // Coefficient::Eval is non-const
      loc_has = 1;
      break;
    }
  }

  double glob_val = 0.0;
  int    glob_has = 0;
  MPI_Allreduce(&loc_val, &glob_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_has, &glob_has, 1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

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

  // center
  try_add(c0);

  // small ring (x–y in 2D, x–z in 3D)
  const int dim = gf.ParFESpace()->GetParMesh()->Dimension();
  for (int k = 0; k < n; ++k) {
    const double th = 2.0 * M_PI * k / n;
    Vector x(c0.Size()); x = c0;
    if (dim == 2) { x[0] += r * cos(th); x[1] += r * sin(th); }
    else          { x[0] += r * cos(th); x[2] += r * sin(th); }
    try_add(x);
  }

  double glob_sum = 0.0; int glob_cnt = 0;
  MPI_Allreduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_cnt, &glob_cnt, 1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
  return (glob_cnt > 0) ? (glob_sum / glob_cnt) : 0.0;
}

// global relative L2 on elements with attribute == small_attr
static double ComputeRelL2_OnAttribute(const ParGridFunction &phi,
                                       const ParGridFunction &exact_gf,
                                       ParMesh &pmesh, int small_attr)
{
  Array<int> elems;
  elems.Reserve(pmesh.GetNE());
  for (int e = 0; e < pmesh.GetNE(); ++e) {
    if (pmesh.GetAttribute(e) == small_attr) elems.Append(e);
  }

  GridFunctionCoefficient exact_coef(&exact_gf);
  ConstantCoefficient zero_c(0.0);

  const double num = phi.ComputeL2Error(exact_coef, /*irs*/nullptr, &elems);
  const double den = exact_gf.ComputeL2Error(zero_c, /*irs*/nullptr, &elems);

  return (den > 0.0) ? (num / den) : 0.0;
}

// for residual visualization on submesh
struct RelErrSignedCoefficient : public Coefficient {
  const ParGridFunction &phi, &exact; double eps;
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

// Save mesh/fields restricted to the small-domain submesh (attribute == small_attr)
static void SaveOnSmallDomainSubmesh(const std::string &tag, int order, int small_attr,
                                     ParMesh &pmesh,
                                     const ParGridFunction &phi_shifted,
                                     const ParGridFunction &exact_shifted,
                                     double relL2_small)
{
  const int world_rank = Mpi::WorldRank();

  // Create submesh containing only 'small_attr'
  Array<int> marker(pmesh.attributes.Max()); marker = 0;
  marker[small_attr - 1] = 1;
  ParSubMesh psub = ParSubMesh::CreateFromDomain(pmesh, marker);
  psub.SetCurvature(std::max(1, order), false, psub.Dimension(), Ordering::byVDIM);

  // FE space and transfers on the submesh
  H1_FECollection sfec(order, psub.Dimension());
  ParFiniteElementSpace sfes(&psub, &sfec);

  ParGridFunction phi_sm(&sfes), exact_sm(&sfes);
  phi_sm = 0.0; exact_sm = 0.0;
  psub.Transfer(phi_shifted,  phi_sm);
  psub.Transfer(exact_shifted, exact_sm);

  // Residual fields for visualization (no L2 norm computation here)
  ParGridFunction resid_sm(&sfes);
  RelErrSignedCoefficient rcoef(phi_sm, exact_sm, 1e-14);
  resid_sm.ProjectCoefficient(rcoef);

  ParGridFunction resid_logabs_sm(&sfes);
  LogAbsCoefficient logabs_coef(resid_sm, 1e-14);
  resid_logabs_sm.ProjectCoefficient(logabs_coef);

  // Build compact writer communicator: only ranks with submesh elements participate.
  const int has_local = (psub.GetNE() > 0) ? 1 : 0;

  MPI_Comm subcomm = MPI_COMM_NULL;
  const int color = has_local ? 1 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subcomm);

  int subrank = -1, subsize = 0;
  if (has_local) {
    MPI_Comm_rank(subcomm, &subrank);
    MPI_Comm_size(subcomm, &subsize);
  }

  // Ensure output dir exists
  if (world_rank == 0) { std::filesystem::create_directories("results"); }

  // Write files only from writers, using compacted suffixes
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

    // Emit a tiny info file + console hint with the exact GLVis command
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

  // Keep relL2 text on world rank 0 (value provided by caller)
  if (Mpi::WorldRank() == 0) {
    std::ofstream rL2("results/" + tag + ".sm.relL2.txt");
    rL2 << std::setprecision(16)
        << "relative_L2_error_on_small_domain " << relL2_small << "\n";
  }
}

} // namespace

// ---------------- main ----------------

int main(int argc, char *argv[])
{
  auto T0 = clk::now();
  Mpi::Init(argc, argv);
  Hypre::Init();

  const int myid   = Mpi::WorldRank();
  const int nprocs = Mpi::WorldSize();

  const char *mesh_file = "mesh/simple_3d_ref.msh";
  int order = 2;
  int small_attr = 1;
  int large_attr = 2;

  double rho_small_dim = 5000.0;
  double rho_large_dim = 0.0;
  double G_dim = 6.67430e-11;

  double L_scale   = 7e6;
  double RHO_scale = rho_small_dim;
  double T_scale   = 1.0 / std::sqrt(G_dim * RHO_scale);

  bool visualization       = false;
  bool dimensional_output  = false;

  // Inner sphere radius (default 0.7)
  double inner_radius = 0.7;

  // ---- CLI ----
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Gmsh mesh file (.msh).");
  args.AddOption(&order, "-o", "--order", "H1 FE order.");
  args.AddOption(&small_attr, "--attr-small", "-as", "Attribute id of small region (>=1).");
  args.AddOption(&large_attr, "--attr-large", "-al", "Attribute id of large region (>=1).");
  args.AddOption(&rho_small_dim, "--rho-small", "-rs", "Dimensional rho in small region.");
  args.AddOption(&rho_large_dim, "--rho-large", "-rl", "Dimensional rho in large region.");
  args.AddOption(&G_dim, "--G", "-G", "Gravitational constant.");
  args.AddOption(&L_scale, "--L", "-L", "Length scale.");
  args.AddOption(&T_scale, "--T", "-T", "Time scale.");
  args.AddOption(&RHO_scale, "--RHO", "-RHO", "Density scale.");
  args.AddOption(&inner_radius, "-ir", "--inner-radius", "Inner sphere radius a (default 0.7).");
  args.AddOption(&dimensional_output,
                 "-dim", "--dimensional-output",
                 "-no-dim", "--no-dimensional-output",
                 "Write phi/exact in dimensional units.", false);
  args.AddOption(&visualization, "-vis", "--visualization",
                 "-no-vis", "--no-visualization", "Send solution to GLVis.");
  args.Parse();
  if (!args.Good()) { if (myid==0) args.PrintUsage(cout); return 1; }
  if (myid==0) args.PrintOptions(cout);

  auto since = [](clk::time_point a){ return std::chrono::duration<double>(clk::now()-a).count(); };
  auto Tsetup0 = clk::now();

  // ---- Scaling ----
  Nondimensionalisation nd(L_scale, T_scale, RHO_scale);
  const double rho_small_nd = nd.ScaleDensity(rho_small_dim);
  const double rho_large_nd = nd.ScaleDensity(rho_large_dim);
  const double G_nd         = G_dim * T_scale * T_scale * RHO_scale;

  // ---- Mesh / FES ----
  Mesh mesh(mesh_file, 1, 1, true);
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();

  const int dim = pmesh.Dimension();

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);

  Array<int> ess_tdof_list;
  if (pmesh.bdr_attributes.Size()) {
    Array<int> ess_bdr(pmesh.bdr_attributes.Max()); ess_bdr = 0;
    pmesh.MarkExternalBoundaries(ess_bdr);
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  auto Tsetup = since(Tsetup0);

  // ---- RHS: -4πG ρ ----
  Vector rho_vals(pmesh.attributes.Max()); rho_vals = 0.0;
  rho_vals(small_attr - 1) = rho_small_nd;
  rho_vals(large_attr - 1) = rho_large_nd;
  PWConstCoefficient rho_pw(rho_vals);

  ConstantCoefficient minus_four_pi_G(-4.0 * M_PI * G_nd);
  ProductCoefficient rhs_coeff(minus_four_pi_G, rho_pw);

  auto Tasm0 = clk::now();

  ParLinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
  b.Assemble();

  ParGridFunction phi(&fes); phi = 0.0;

  ConstantCoefficient one(1.0);
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();

  auto Tasm = since(Tasm0);

  // ---- Solve ----
  auto Tsolve0 = clk::now();

  OperatorPtr A; Vector B, X;
  a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);

  HypreBoomerAMG amg; amg.SetOperator(*A);
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(5000);
  cg.SetPrintLevel(myid==0 ? 1 : 0);
  cg.SetPreconditioner(amg);
  cg.SetOperator(*A);
  cg.Mult(B, X);

  a.RecoverFEMSolution(X, b, phi);

  auto Tsolve = since(Tsolve0);

  // ---- Post ----
  auto Tpost0 = clk::now();

  // Centroid of SMALL region (attribute small_attr)
  Array<int> small_marker(pmesh.attributes.Max()); small_marker = 0;
  small_marker[small_attr - 1] = 1;
  Vector c0 = mfemElasticity::MeshCentroid(&pmesh, small_marker, /*order*/1);

  // Adaptive sampling radius: min element "diameter" on the small region
  double h_small = std::numeric_limits<double>::infinity();
  for (int e = 0; e < pmesh.GetNE(); ++e) {
    if (pmesh.GetAttribute(e) != small_attr) continue;
    h_small = std::min(h_small, pmesh.GetElementSize(e, 1)); // 1 = diameter-like metric
  }
  double h_small_glob = h_small;
  MPI_Allreduce(&h_small, &h_small_glob, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  const double r_eps = 0.3 * h_small_glob; // assumed finite; no lower guard
  const int    nsmp  = 24;

  // Subtract gauge from numerical solution via ring average at centroid
  const double phi_c = SampledValueAtGlobal(phi, c0, r_eps, nsmp);
  phi -= phi_c;

  // Analytic solution coefficient for a uniform sphere of radius 'inner_radius'
  UniformSphereSolution base(dim, c0, inner_radius);
  FunctionCoefficient base_exact = base.Coefficient();

  // Scale analytic coefficient by rho*G using a named ConstantCoefficient (lvalue)
  ConstantCoefficient scale_pos(rho_small_nd * G_nd);
  ProductCoefficient exact_coeff_scaled(scale_pos, base_exact);

  // Project analytic to a gridfunction for L2 norms/plots
  ParGridFunction exact_gf(&fes); exact_gf = 0.0;
  exact_gf.ProjectCoefficient(exact_coeff_scaled);

  // Subtract gauge from analytic solution by evaluating the coefficient exactly at centroid
  const double exact_c0 = EvalCoefficientAtPointGlobal(exact_coeff_scaled, pmesh, c0);
  exact_gf -= exact_c0;

  // Shift (+1) if you want strictly positive outputs (optional; keeps parity with older scripts)
  ParGridFunction phi_out(&fes), exact_out(&fes);
  phi_out = phi; exact_out = exact_gf;
  phi_out  += 1.0;
  exact_out += 1.0;

  if (dimensional_output) {
    nd.UnscaleGravityPotential(phi_out);
    nd.UnscaleGravityPotential(exact_out);
  }

  // Global relative L2 on inner sphere (attribute == small_attr)
  const std::string tag = std::filesystem::path(mesh_file).stem().string();
  std::filesystem::create_directories("results");

  const double relL2_small = ComputeRelL2_OnAttribute(phi_out, exact_out, pmesh, small_attr);
  if (Mpi::WorldRank() == 0) {
    std::ofstream rL2("results/" + tag + ".sm.relL2.txt");
    rL2 << std::setprecision(16)
        << "relative_L2_error_on_small_domain " << relL2_small << "\n";
    std::cout << std::scientific << std::setprecision(6)
              << "[relL2] inner-sphere = " << relL2_small << std::endl;
  }

  // I/O timing (submesh save + small-domain files)
  auto Tio0 = clk::now();
  SaveOnSmallDomainSubmesh(tag, order, small_attr, pmesh, phi_out, exact_out, relL2_small);
  const double Tio = since(Tio0);

  auto Tpost = since(Tpost0);

  if (myid == 0) {
    ofstream times("results/" + tag + ".timings.txt");
    const double Ttot = std::chrono::duration<double>(clk::now() - T0).count();
    times << std::fixed << std::setprecision(6)
          << "setup   " << Tsetup << "\n"
          << "assemble " << Tasm  << "\n"
          << "solve    " << Tsolve<< "\n"
          << "postproc " << Tpost << "\n"
          << "io       " << Tio   << "\n"
          << "total    " << Ttot  << "\n";
    cout << "[timings] setup=" << Tsetup << "s assemble=" << Tasm
         << "s solve=" << Tsolve << "s postproc=" << Tpost
         << "s io=" << Tio << "s total=" << Ttot << "s\n";
  }

  if (visualization) {
    socketstream ss("localhost", 19916);
    ss << "parallel " << nprocs << " " << myid << "\n";
    ss.precision(8);
    ss << "solution\n" << pmesh << phi << flush;
    if (dim == 2) ss << "keys Roc0l\n" << flush; else ss << "keys RRRilmc\n" << flush;
  }
  return 0;
}

