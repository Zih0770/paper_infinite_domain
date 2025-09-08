// src/Poisson_multipole.cpp
#include "mfem.hpp"
#include <mpi.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "uniform_sphere.hpp"
#include "common.hpp"

// --- mfemElasticity (provides PoissonMultipoleOperator) ---
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace std;
using namespace mfemElasticity;
using clk = std::chrono::steady_clock;

// ===================== Helpers (full implementations) =====================

// Compute centroid of a given volume attribute (parallel reduction)
static Vector ComputeRegionCentroid(ParMesh &pmesh, int attr)
{
  const int dim = pmesh.Dimension();
  Vector s(dim); s = 0.0; double w = 0.0;
  for (int e = 0; e < pmesh.GetNE(); ++e) {
    if (pmesh.GetAttribute(e) != attr) continue;
    Vector c(dim); pmesh.GetElementCenter(e, c);
    const double ve = pmesh.GetElementVolume(e);
    for (int d = 0; d < dim; ++d) s[d] += ve * c[d];
    w += ve;
  }
  Vector sg(dim); sg = 0.0; double wg = 0.0;
  MPI_Allreduce(s.GetData(), sg.GetData(), dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&w, &wg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (wg > 0.0) sg /= wg;
  return sg;
}

// Global sampling of a ParGridFunction at a small ring + center,
// with (sum,count) MPI reductions
static double SampledValueAtGlobal(const ParGridFunction &gf, const Vector &c0, double r, int n)
{
  double loc_sum = 0.0; int loc_cnt = 0;

  auto try_add = [&](const Vector &x){
    auto *pmesh = gf.ParFESpace()->GetParMesh();
    for (int e = 0; e < pmesh->GetNE(); ++e) {
      ElementTransformation *T = pmesh->GetElementTransformation(e); // <-- FIXED: '->'
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

// L2 norm^2 via quadrature over all local elements, then MPI-reduce
static double L2NormSq_Quadrature(const ParFiniteElementSpace &fes, const ParGridFunction &u)
{
  double loc = 0.0;
  const int NE = fes.GetNE();
  for (int e = 0; e < NE; ++e)
  {
    ElementTransformation *T = fes.GetElementTransformation(e);
    const FiniteElement *fe  = fes.GetFE(e);
    const Geometry::Type gt  = fe->GetGeomType();
    const int qorder = 2*fe->GetOrder() + 2;
    const IntegrationRule &ir = IntRules.Get(gt, qorder);
    for (int i = 0; i < ir.GetNPoints(); ++i)
    {
      const IntegrationPoint &ip = ir.IntPoint(i);
      T->SetIntPoint(&ip);
      const double v = u.GetValue(*T, ip);
      loc += v*v * ip.weight * T->Weight();
    }
  }
  double glob = 0.0;
  MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return glob;
}

// (phi - exact) / (|exact| + eps)
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

// log10(|field| + eps)
struct LogAbsCoefficient : public Coefficient {
  const ParGridFunction &f; double eps;
  LogAbsCoefficient(const ParGridFunction &ff, double e=1e-14) : f(ff), eps(e) {}
  double Eval(ElementTransformation &T, const IntegrationPoint &ip) override {
    const double v = f.GetValue(T, ip);
    return std::log10(std::abs(v) + eps);
  }
};

// Estimate inner spherical interface radius from the small_attr submesh boundary
static std::pair<int,double>
EstimateInnerSphereRadius(ParMesh &pmesh, int small_attr, const Vector &c0)
{
  Array<int> marker(pmesh.attributes.Max()); marker = 0;
  marker[small_attr - 1] = 1;
  ParSubMesh psub = ParSubMesh::CreateFromDomain(pmesh, marker);

  ParMesh &sm = psub;
  const int dim = sm.Dimension();

  Array<int> bdr_vtx(sm.GetNV()); bdr_vtx = 0;
  for (int be = 0; be < sm.GetNBE(); ++be) {
    Array<int> v;
    sm.GetBdrElementVertices(be, v);
    for (int i = 0; i < v.Size(); ++i) bdr_vtx[v[i]] = 1;
  }

  double loc_sum = 0.0; int loc_cnt = 0;
  for (int vi = 0; vi < sm.GetNV(); ++vi) {
    if (!bdr_vtx[vi]) continue;
    const double *vx = sm.GetVertex(vi);
    Vector x(dim);
    for (int d = 0; d < dim; ++d) x[d] = vx[d];
    double r2 = 0.0;
    for (int d = 0; d < dim; ++d) { const double dx = x[d] - c0[d]; r2 += dx*dx; }
    loc_sum += std::sqrt(r2);
    loc_cnt += 1;
  }

  double glob_sum = 0.0; int glob_cnt = 0;
  MPI_Allreduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_cnt, &glob_cnt, 1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

  if (glob_cnt == 0) return {0, 0.0};
  return {1, glob_sum / glob_cnt};
}

// Write results restricted to the "small domain" submesh (attribute small_attr)
static void SaveOnSmallDomainSubmesh(const std::string &tag, int order, int small_attr,
                                     ParMesh &pmesh,
                                     const ParGridFunction &phi_shifted,
                                     const ParGridFunction &exact_shifted)
{
  const int world_rank = Mpi::WorldRank();

  // Create submesh containing only 'small_attr'
  Array<int> marker(pmesh.attributes.Max()); marker = 0;
  marker[small_attr - 1] = 1;
  ParSubMesh psub = ParSubMesh::CreateFromDomain(pmesh, marker);
  psub.SetCurvature(std::max(1, order), false, psub.Dimension(), Ordering::byVDIM);

  // FE space and field transfers on the submesh
  H1_FECollection sfec(order, psub.Dimension());
  ParFiniteElementSpace sfes(&psub, &sfec);

  ParGridFunction phi_sm(&sfes), exact_sm(&sfes);
  phi_sm = 0.0; exact_sm = 0.0;
  psub.Transfer(phi_shifted,  phi_sm);
  psub.Transfer(exact_shifted, exact_sm);

  // residual (signed relative error)
  ParGridFunction resid_sm(&sfes);
  RelErrSignedCoefficient rcoef(phi_sm, exact_sm, 1e-14);
  resid_sm.ProjectCoefficient(rcoef);

  // log10|residual|
  ParGridFunction resid_logabs_sm(&sfes);
  LogAbsCoefficient logabs_coef(resid_sm, 1e-14);
  resid_logabs_sm.ProjectCoefficient(logabs_coef);

  // relative L2 error
  ParGridFunction diff_sm(&sfes); diff_sm = phi_sm; diff_sm -= exact_sm;
  const double num_sq = L2NormSq_Quadrature(sfes, diff_sm);
  const double den_sq = L2NormSq_Quadrature(sfes, exact_sm);
  const double relL2  = (den_sq > 0.0) ? std::sqrt(num_sq/den_sq) : 0.0;

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

    // Emit info + console hint with GLVis command
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

  // Keep relL2 text on world rank 0
  if (Mpi::WorldRank() == 0) {
    std::ofstream rL2("results/" + tag + ".sm.relL2.txt");
    rL2 << std::setprecision(16)
        << "relative_L2_error_on_small_domain " << relL2 << "\n";
  }
}

// =============================== main =====================================

int main(int argc, char *argv[])
{
  auto T0 = clk::now();
  Mpi::Init(argc, argv);
  Hypre::Init();

  const int myid   = Mpi::WorldRank();
  const int nprocs = Mpi::WorldSize();

  const char *mesh_file = "mesh/simple_2d_4.msh";
  int order = 2;
  int lmax  = 4;   // Multipole cutoff (ℓ_max)
  int small_attr = 1;
  int large_attr = 2;

  double rho_small_dim = 5514.0;
  double rho_large_dim = 0.0;
  double G_dim         = 6.67430e-11;

  double L_scale   = 1.0;
  double RHO_scale = rho_small_dim;
  double T_scale   = 1.0 / std::sqrt(G_dim * RHO_scale);

  bool visualization = false;
  bool dimensional_output = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Gmsh mesh file (.msh).");
  args.AddOption(&order, "-o", "--order", "H1 FE order.");
  args.AddOption(&lmax, "-lmax", "--lmax", "Multipole spherical-harmonic cutoff (ℓmax).");
  args.AddOption(&small_attr, "--attr-small", "-as", "Attribute id of small region (>=1).");
  args.AddOption(&large_attr, "--attr-large", "-al", "Attribute id of large region (>=1).");
  args.AddOption(&rho_small_dim, "--rho-small", "-rs", "Dimensional rho in small region.");
  args.AddOption(&rho_large_dim, "--rho-large", "-rl", "Dimensional rho in large region.");
  args.AddOption(&G_dim, "--G", "-G", "Gravitational constant.");
  args.AddOption(&L_scale, "--L", "-L", "Length scale.");
  args.AddOption(&T_scale, "--T", "-T", "Time scale.");
  args.AddOption(&RHO_scale, "--RHO", "-RHO", "Density scale.");
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

  // ------------------------------ Setup -----------------------------------
  auto Tsetup0 = clk::now();

  Nondimensionalisation nd(L_scale, T_scale, RHO_scale);
  const double rho_small_nd = nd.ScaleDensity(rho_small_dim);
  const double rho_large_nd = nd.ScaleDensity(rho_large_dim);
  const double G_nd         = G_dim * T_scale * T_scale * RHO_scale;

  Mesh mesh(mesh_file, 1, 1, true);
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();

  const int dim = pmesh.Dimension();
  pmesh.SetCurvature(std::max(1, order), false, dim, Ordering::byVDIM);

  H1_FECollection      fec(order, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);

  // External boundary marker (used in 2D net-flux correction)
  Array<int> ext_bdr;
  if (pmesh.bdr_attributes.Size()) {
    ext_bdr.SetSize(pmesh.bdr_attributes.Max());
    ext_bdr = 0;
    pmesh.MarkExternalBoundaries(ext_bdr);
  }

  // No essential BCs
  Array<int> ess_tdof_list;

  auto Tsetup = since(Tsetup0);

  // ------------------------------ Assembly --------------------------------
  Vector rho_vals(pmesh.attributes.Max()); rho_vals = 0.0;
  rho_vals(small_attr - 1) = rho_small_nd;
  rho_vals(large_attr - 1) = rho_large_nd;
  PWConstCoefficient rho_pw(rho_vals);

  ConstantCoefficient minus_four_pi_G(-4.0 * M_PI * G_nd);
  ProductCoefficient rhs_coeff(minus_four_pi_G, rho_pw);

  auto Tasm0 = clk::now();

  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator);
  a.Assemble();

  // mass shift for simple smoother preconditioning
  ConstantCoefficient eps(0.01);
  ParBilinearForm a_shift(&fes, &a);
  a_shift.AddDomainIntegrator(new MassIntegrator(eps));
  a_shift.Assemble();
  a_shift.Finalize();  // required before ParallelAssemble

  ParLinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
  b.Assemble();

  // 2D tweak: remove net source using uniform boundary load on external boundary
  if (dim == 2 && pmesh.bdr_attributes.Size()) {
    ParGridFunction x1(&fes); x1 = 1.0;

    // total mass from domain RHS
    double mass = b(x1);

    // external boundary length
    ParLinearForm bb(&fes);
    ConstantCoefficient one(1.0);
    bb.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), ext_bdr);
    bb.Assemble();
    double L = bb(x1);

    if (L != 0.0) { b.Add(-mass / L, bb); }
  }

  auto Tasm = since(Tasm0);

  // -------------- Multipole: operator build + RHS correction ---------------
  auto Tmulti0 = clk::now();

  // Discontinuous (L2) space for projecting density for the multipole operator
  L2_FECollection dfec(std::max(0, order - 1), dim);
  ParFiniteElementSpace dfes(&pmesh, &dfec);

  // Mark the *inner* domain (attribute small_attr) for the multipole source
  Array<int> inner_marker(pmesh.attributes.Max());
  inner_marker = 0;
  inner_marker[small_attr - 1] = 1;

  // Build multipole operator; degree controlled by -lmax
  PoissonMultipoleOperator C(MPI_COMM_WORLD, &dfes, &fes, lmax, inner_marker);

  // Project physical density (unscaled) to L2 and apply correction to RHS.
  // b currently contains (-4π G_nd) ∫ ρ v; add (+4π G_nd) * C[ρ] to match continuum scaling.
  ParGridFunction z(&dfes);
  z.ProjectCoefficient(rho_pw);         // unscaled density
  C.AddMult(z, b, +4.0 * M_PI * G_nd);  // add the exterior correction

  auto Tmulti = since(Tmulti0);

  // ----------------------------- Solve ------------------------------------
  auto Tsolve0 = clk::now();

  ParGridFunction phi(&fes); phi = 0.0;

  OperatorPtr A; Vector B, X;
  a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);

  // Preconditioner on mass-shifted stiffness
  HypreParMatrix *As = a_shift.ParallelAssemble();
  HypreBoomerAMG P(*As);

  CGSolver cg(MPI_COMM_WORLD);
  cg.SetOperator(*A);
  cg.SetPreconditioner(P);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(10000);
  cg.SetPrintLevel(myid==0 ? 1 : 0);

  // IMPORTANT: project out constant nullspace in both 2D and 3D
  OrthoSolver ortho(MPI_COMM_WORLD);
  ortho.SetSolver(cg);
  ortho.Mult(B, X);

  a.RecoverFEMSolution(X, b, phi);

  auto Tsolve = since(Tsolve0);

  // --------------------------- Postprocessing ------------------------------
  auto Tpost0 = clk::now();

  Vector c0 = ComputeRegionCentroid(pmesh, small_attr);

  // Always shift both numerical and analytical fields so the value at c0 is 1.0
  const double r_eps = 0.02; const int nsmp = 12;
  double phi_c = SampledValueAtGlobal(phi, c0, r_eps, nsmp);
  phi -= phi_c;

  // Exact interior (uniform sphere) with measured inner radius r1
  auto [r_found, r1] = EstimateInnerSphereRadius(pmesh, small_attr, c0);
  if (!r_found) { r1 = 1.0; }

  UniformSphereSolution base(dim, c0, r1);
  FunctionCoefficient base_exact = base.Coefficient();
  ConstantCoefficient scale_pos(rho_small_nd * G_nd);
  ProductCoefficient exact_coeff_scaled(scale_pos, base_exact);

  ParGridFunction exact_gf(&fes); exact_gf = 0.0;
  exact_gf.ProjectCoefficient(exact_coeff_scaled);

  double exact_c0 = SampledValueAtGlobal(exact_gf, c0, r_eps, nsmp);
  exact_gf -= exact_c0;

  // Offset so GLVis bars are positive and center value is +1
  ParGridFunction phi_out(&fes), exact_out(&fes);
  phi_out = phi; exact_out = exact_gf;
  if (dimensional_output) { nd.UnscaleGravityPotential(phi_out); nd.UnscaleGravityPotential(exact_out); }
  phi_out  += 1.0;
  exact_out += 1.0;

  auto Tpost = since(Tpost0);

  // ------------------------------- I/O ------------------------------------
  std::filesystem::create_directories("results");

  // Include lmax in tag
  std::ostringstream tag_ss;
  tag_ss << std::filesystem::path(mesh_file).stem().string()
         << "_Multipole_lmax" << lmax;
  const std::string tag = tag_ss.str();

  auto Tio0 = clk::now();
  SaveOnSmallDomainSubmesh(tag, order, small_attr, pmesh, phi_out, exact_out);
  auto Tio = since(Tio0);

  if (myid == 0) {
    ofstream times("results/" + tag + ".timings.txt");
    const double Ttot = std::chrono::duration<double>(clk::now() - T0).count();
    times << std::fixed << std::setprecision(6)
          << "assemble "           << Tasm   << "\n"
          << "Multipole_assemble " << Tmulti << "\n"
          << "solve    "           << Tsolve << "\n"
          << "postproc "           << Tpost  << "\n"
          << "io       "           << Tio    << "\n"
          << "total    "           << Ttot   << "\n";
    cout << "[timings] assemble=" << Tasm
         << "s Multipole_assemble=" << Tmulti
         << "s solve="              << Tsolve
         << "s postproc="           << Tpost
         << "s io="                 << Tio
         << "s total="              << Ttot << "s\n";
  }

  if (visualization) {
    socketstream ss("localhost", 19916);
    ss << "parallel " << nprocs << " " << myid << "\n";
    ss.precision(8);
    ss << "solution\n" << pmesh << phi << flush;
    if (dim == 2) ss << "keys oc0l\n" << flush;
    else          ss << "keys ilmc\n" << flush;
  }

  delete As;  // from a_shift.ParallelAssemble()

  return 0;
}

