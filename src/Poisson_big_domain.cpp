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

static double ComputeRelL2_OnAttribute(const ParGridFunction &phi,
                                       const ParGridFunction &exact_gf,
                                       ParMesh &pmesh, int small_attr_idx)
{
  // small_attr_idx is 0-based: 0 -> attribute ID 1, 1 -> attribute ID 2, ...
  Array<int> attr_marker(pmesh.attributes.Max());
  attr_marker = 0;

  MFEM_ASSERT(small_attr_idx >= 0 && small_attr_idx < attr_marker.Size(),
              "small_attr index out of range.");
  attr_marker[small_attr_idx] = 1;

  GridFunctionCoefficient exact_coef(&exact_gf);
  ConstantCoefficient zero_c(0.0);

  const double num = phi.ComputeL2Error(exact_coef, /*irs*/nullptr, &attr_marker);
  const double den = exact_gf.ComputeL2Error(zero_c,    /*irs*/nullptr, &attr_marker);

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
  marker[small_attr] = 1;
  ParSubMesh psub = ParSubMesh::CreateFromDomain(pmesh, marker);

  // FE space and transfers on the submesh
  H1_FECollection sfec(order, psub.Dimension());
  ParFiniteElementSpace sfes(&psub, &sfec);

  ParGridFunction phi_sm(&sfes), exact_sm(&sfes);
  phi_sm = 0.0; exact_sm = 0.0;
  psub.Transfer(phi_shifted,  phi_sm);
  psub.Transfer(exact_shifted, exact_sm);

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
  int small_attr = 0;
  int large_attr = 1;

  double rho_small_dim = 5000.0;
  double rho_large_dim = 0.0;
  double G_dim = 6.67430e-11;

  double L_scale   = 7e6;
  double RHO_scale = rho_small_dim;
  double T_scale   = 1.0 / std::sqrt(G_dim * RHO_scale);

  bool visualization       = false;
  bool dimensional_output  = false;

  enum BcType { BC_DIRICHLET = 0, BC_NEUMANN = 1 };
  int bc = BC_DIRICHLET;

  // ---- CLI ----
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Gmsh mesh file (.msh).");
  args.AddOption(&order, "-o", "--order", "H1 FE order.");
  args.AddOption(&bc, "-bc", "--bc",
               "Boundary condition: 0 = Dirichlet(0) on outer boundary, "
               "1 = Neumann(0) (natural/no boundary treatment).");
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
  auto Tsetup0 = clk::now();

  // ---- Scaling ----
  Nondimensionalisation nd(L_scale, T_scale, RHO_scale);
  const double rho_small_nd = nd.ScaleDensity(rho_small_dim);
  const double rho_large_nd = nd.ScaleDensity(rho_large_dim);
  const double G_nd = G_dim * T_scale * T_scale * RHO_scale;
  real_t c = -4.0 * M_PI * G_dim * RHO_scale * T_scale * T_scale;
  if (myid == 0)
  {
      cout<<"RHS dimensionless number c = "<<c<<endl;
  }

  // ---- Mesh / FES ----
  Mesh mesh(mesh_file, 1, 1, true);
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();

  const int dim = pmesh.Dimension();
  if (myid==0)
  {
      std::cout << "num volume attrs = " << pmesh.attributes.Size() << "\n";
      std::cout << "num bdr attrs    = " << pmesh.bdr_attributes.Size() << "\n";
  }

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) { cout << "Global true dofs: " << size << endl; }

  Array<int> ess_tdof_list;
  if (bc == BC_DIRICHLET && pmesh.bdr_attributes.Size()) {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max()); ess_bdr = 0;
      pmesh.MarkExternalBoundaries(ess_bdr);
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  auto Tsetup = since(Tsetup0);

  // ---- RHS: -4πG rho ----
  Vector rho_vals(pmesh.attributes.Max()); rho_vals = 0.0;
  rho_vals(small_attr) = rho_small_nd;
  rho_vals(large_attr) = rho_large_nd;
  PWConstCoefficient rho_pw(rho_vals);

  ProductCoefficient rhs_coeff(c, rho_pw);

  auto Tasm0 = clk::now();

  ParLinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
  b.Assemble(); 

  ConstantCoefficient one(1.0);
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();

  /*if (bc == BC_NEUMANN) {
      ParLinearForm L(&fes);
      ConstantCoefficient one_c(1.0);
      L.AddDomainIntegrator(new DomainLFIntegrator(one_c));
      L.Assemble();

      ParGridFunction z(&fes); z = 1.0;
      const double volume = L(z);   
      const double f_int  = b(z);  

      if (volume != 0.0) {
          b.Add(-(f_int/volume), L);
      }
  }*/

  auto Tasm = since(Tasm0);

  // ---- Solve ----
  auto Tsolve0 = clk::now();

  ParGridFunction phi(&fes); phi = 0.0;
  OperatorPtr A; Vector B, X;
  a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);

  HypreBoomerAMG amg; amg.SetOperator(*A);
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(5000);
  cg.SetPrintLevel(myid==0 ? 1 : 0);
  cg.SetPreconditioner(amg);
  cg.SetOperator(*A);

  if (bc == BC_NEUMANN) {
      OrthoSolver ortho(MPI_COMM_WORLD);
      ortho.SetSolver(cg);      
      ortho.Mult(B, X);         
  } else {
      cg.Mult(B, X);       
  }

  a.RecoverFEMSolution(X, b, phi);

  auto Tsolve = since(Tsolve0);

  // ---- Post ----
  auto Tpost0 = clk::now();

  // Centroid of SMALL region (attribute small_attr)
  Array<int> v1_marker(pmesh.attributes.Max()); v1_marker = 0;
  v1_marker[small_attr] = 1;
  Vector c0 = mfemElasticity::MeshCentroid(&pmesh, v1_marker, /*order*/1);
  if (myid == 0) { cout << "centroid (small region): "; c0.Print(cout); }

  Array<int> b1_marker(pmesh.bdr_attributes.Max());
  b1_marker = 0;
  b1_marker[11-1] = 1;
  auto [found0, same0, r0] = mfemElasticity::SphericalBoundaryRadius(&pmesh, b1_marker, c0);
  if (myid == 0) { std::cout<<"r0: "<<r0<<std::endl; }

  // Analytic solution coefficient for a uniform sphere of radius 'inner_radius'
  UniformSphereSolution base(dim, c0, r0);
  FunctionCoefficient base_exact = base.Coefficient();

  // Scale analytic coefficient by rho*G using a named ConstantCoefficient (lvalue)
  //ConstantCoefficient scale_pos(rho_small_nd * G_nd);
  ConstantCoefficient scale_pos(-rho_small_nd * c / 4.0 / M_PI);
  ProductCoefficient exact_coeff_scaled(scale_pos, base_exact);

  // Project analytic to a gridfunction for L2 norms/plots
  ParGridFunction exact_gf(&fes); exact_gf = 0.0;
  exact_gf.ProjectCoefficient(exact_coeff_scaled);

  auto phi_coef = GridFunctionCoefficient(&phi);

  // Subtract gauge from analytic solution by evaluating the coefficient exactly at centroid
  const double phi_c0 = EvalCoefficientAtPointGlobal(phi_coef, pmesh, c0);
  const double exact_c0 = EvalCoefficientAtPointGlobal(exact_coeff_scaled, pmesh, c0);
  if (Mpi::WorldRank() == 0) {
      std::cout << std::setprecision(12)
          << "[centre] phi(center)   = " << phi_c0 << "\n"
          << "[centre] exact(center) = " << exact_c0 << std::endl;
  }
  phi -= phi_c0;
  exact_gf -= exact_c0;

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

