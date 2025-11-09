// src/ex2p_ext.cpp
#include "mfem.hpp"
#include "mfemElasticity.hpp"

#include "common.hpp"
#include "uniform_sphere.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = std::numbers::pi_v<real_t>;

static inline real_t lerp_clamped(real_t x, const std::vector<real_t>& X,
                                  const std::vector<real_t>& Y)
{
    const int n = (int)X.size();
    if (n == 0) { return 0.0; }
    if (n == 1) { return Y[0]; }
    if (x <= X.front()) { return Y.front(); }
    if (x >= X.back())  { return Y.back(); }

    auto it = std::lower_bound(X.begin(), X.end(), x); // first X[i] >= x
    int i1 = int(it - X.begin());
    int i0 = i1 - 1;
    real_t x0 = X[i0], x1 = X[i1];
    real_t y0 = Y[i0], y1 = Y[i1];
    real_t t  = (x - x0) / (x1 - x0);
    return (1.0 - t) * y0 + t * y1;
}

class RadialInterpCoefficient : public Coefficient
{
private:
    std::vector<real_t> r_dim_;
    std::vector<real_t> rho_dim_;
    real_t L_;
public:
    RadialInterpCoefficient(std::vector<real_t> r_dim,
                            std::vector<real_t> rho_dim,
                            real_t L)
        : r_dim_(std::move(r_dim)), rho_dim_(std::move(rho_dim)), L_(L) {}

    virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
    {
        Vector x;
        T.Transform(ip, x);
        real_t r_nd = x.Norml2();
        real_t r_dim = r_nd * L_;
        return lerp_clamped(r_dim, r_dim_, rho_dim_);
    }
};

static bool ReadPREMAndSplitLayers(const std::string &filename,
                                   std::vector<std::vector<real_t>> &layers_r_dim,
                                   std::vector<std::vector<real_t>> &layers_rho_dim,
                                   real_t radius_scale = 1.0,
                                   real_t density_scale = 1.0)
{
    std::ifstream in(filename);
    if (!in) { return false; }

    std::string line;
    // Skip first 3 header lines
    for (int i = 0; i < 3; ++i)
    {
        if (!std::getline(in, line)) return false;
    }

    std::vector<real_t> R, RH;
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
 
        std::istringstream iss(line);
        real_t r, rho;
        if (!(iss >> r >> rho)) continue; 

        R.push_back(r * radius_scale);
        RH.push_back(rho * density_scale);
    }

    if (R.empty()) { return false; }

    const real_t tol = 1e-6;
    layers_r_dim.clear();
    layers_rho_dim.clear();

    std::vector<real_t> curR, curRH;
    curR.push_back(R[0]);
    curRH.push_back(RH[0]);

    for (size_t i = 1; i < R.size(); ++i)
    {
        if (std::abs(R[i] - R[i-1]) <= tol)
        {
            if (!curR.empty())
            {
                std::vector<real_t> fR, fRH;
                fR.reserve(curR.size());
                fRH.reserve(curRH.size());
                for (size_t k = 0; k < curR.size(); ++k)
                {
                    if (k == 0 || curR[k] > fR.back()) { fR.push_back(curR[k]); fRH.push_back(curRH[k]); }
                    else if (curR[k] == fR.back())     { fRH.back() = curRH[k]; }
                }
                layers_r_dim.push_back(std::move(fR));
                layers_rho_dim.push_back(std::move(fRH));
            }
            curR.clear(); curRH.clear();
            curR.push_back(R[i]);
            curRH.push_back(RH[i]);
        }
        else
        {
            curR.push_back(R[i]);
            curRH.push_back(RH[i]);
        }
    }

    if (!curR.empty())
    {
        std::vector<real_t> fR, fRH;
        fR.reserve(curR.size());
        fRH.reserve(curRH.size());
        for (size_t k = 0; k < curR.size(); ++k)
        {
            if (k == 0 || curR[k] > fR.back()) { fR.push_back(curR[k]); fRH.push_back(curRH[k]); }
            else if (curR[k] == fR.back())     { fRH.back() = curRH[k]; }
        }
        layers_r_dim.push_back(std::move(fR));
        layers_rho_dim.push_back(std::move(fRH));
    }

    return !layers_r_dim.empty();
}

int main(int argc, char *argv[])
{
  Mpi::Init(argc, argv);
  Hypre::Init();
  const int myid   = Mpi::WorldRank();
  const int nprocs = Mpi::WorldSize();

  const char *mesh_file = "mesh/prem.msh";
  const char *property_file = "data/prem.200.noiso";
  int order = 3;
  int lmax  = 8;   
  int method = 10;
  int linearised = 0;

  real_t G_dim = 6.67430e-11;

  real_t L_scale = 6368000.;
  real_t RHO_scale = 5000.0;
  real_t T_scale = 1.0 / std::sqrt(G_dim * RHO_scale);

  real_t start_time, end_time, assembly_time, solver_time;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Gmsh mesh file (.msh).");
  args.AddOption(&order, "-o", "--order", "H1 FE order.");
  args.AddOption(&lmax, "-lmax", "--lmax", "spherical-harmonic cutoff.");
  args.AddOption(&method, "-mth", "--method",
                 "Solution method: 0 = Neumann, 1 = DtN, 2 = multipole, 10 = Dirichlet.");
  args.AddOption(&linearised, "-lin", "--linearised",
                 "Solve reference (0) or linearised (1) problem.");
  args.AddOption(&L_scale, "--L", "-L", "Length scale.");
  args.AddOption(&T_scale, "--T", "-T", "Time scale.");
  args.AddOption(&RHO_scale, "--RHO", "-RHO", "Density scale.");
  args.Parse();
  if (!args.Good()) { 
      if (myid==0) args.PrintUsage(cout); 
      return 1; }
  if (myid==0) args.PrintOptions(cout);

  std::vector<std::vector<real_t>> layers_r_dim, layers_rho_dim;
  if (!ReadPREMAndSplitLayers(property_file, layers_r_dim, layers_rho_dim, L_scale, RHO_scale))
  {
      std::cerr << "Error: failed to read PREM property file: " << property_file << std::endl;
      return 1;
  }
  const int num_layers = (int)layers_r_dim.size();

  std::vector<std::unique_ptr<RadialInterpCoefficient>> layer_coeffs;
  layer_coeffs.reserve(num_layers);
  for (int i = 0; i < num_layers; ++i)
  {
      layer_coeffs.emplace_back(std::make_unique<RadialInterpCoefficient>(layers_r_dim[i], layers_rho_dim[i], 1.0));
  }

  Nondimensionalisation nd(L_scale, T_scale, RHO_scale);
  const real_t G_nd = G_dim * T_scale * T_scale * RHO_scale;
  real_t G = -4.0 * pi * G_dim * RHO_scale * T_scale * T_scale;
  ConstantCoefficient G_coeff(G);  
  if (myid == 0)
  {
      cout<<"RHS dimensionless number G = "<<G<<endl;
  }

  Mesh mesh(mesh_file, 1, 1, true);
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  const int dim = pmesh.Dimension();

  Array<int> dom_marker(pmesh.attributes.Max()); 
  dom_marker = 0;
  for (int i = 0; i < pmesh.attributes.Max() - 1; i++) {
      dom_marker[i] = 1;
  }

  auto bdr_marker = Array<int>(pmesh.bdr_attributes.Max());
  bdr_marker = 0;
  for (int i = 0; i < pmesh.bdr_attributes.Max() - 1; i++) {
      bdr_marker[i] = 1;
  }

  H1_FECollection fec(order, dim);
  L2_FECollection dfec(order - 1, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);
 
  std::unique_ptr<ParFiniteElementSpace> dfes;
  if (method == 2) {
    dfes = std::make_unique<ParFiniteElementSpace>(&pmesh, &dfec);
  }

  std::unique_ptr<ParFiniteElementSpace> vfes;
  if (linearised == 1) {
    vfes = std::make_unique<ParFiniteElementSpace>(&pmesh, &dfec, dim);
  }

  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) { cout << "Number of finite element unknowns: " << size << endl; }

  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  ConstantCoefficient eps(0.001);
  ParBilinearForm as(&fes);
  as.AddDomainIntegrator(new DiffusionIntegrator());
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  auto rho_coeff = PWCoefficient();
  for (int i = 1; i < num_layers; i++) {
      rho_coeff.UpdateCoefficient(i, *(layer_coeffs[i-1].get()));
  } 
  ProductCoefficient rhs_coeff(G_coeff, rho_coeff);

  auto x = ParGridFunction(&fes);

  ParLinearForm b(&fes);

  auto uv = Vector(dim);
  uv = 1.0;

  std::unique_ptr<ParGridFunction> u; 
  if (linearised == 1) {
      auto uCoeff1 = VectorConstantCoefficient(uv);
      auto uCoeff = PWVectorCoefficient(dim);
      for (int i = 1; i < num_layers; i++) {
          uCoeff.UpdateCoefficient(i, uCoeff1);
      }
      u = std::make_unique<ParGridFunction>(vfes.get());
      u->ProjectCoefficient(uCoeff);
  }

  auto boundary_marker = ExternalBoundaryMarker(&pmesh);
  if (linearised == 0) {
      b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
      b.Assemble();

      if (method == 1 && dim == 2) {
          x = 1.0;
          auto mass = b(x);
          auto l = ParLinearForm(&fes);
          auto one = ConstantCoefficient(1);
          l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
          l.Assemble();
          auto length = l(x);
          b.Add(-mass / length, l);
      }
  } else {
      auto d = ParMixedBilinearForm(&fes, vfes.get());
      d.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rhs_coeff)); 
      d.Assemble();
      d.MultTranspose(*u, b);
  }

  if (method == 2) {
      if (linearised == 0) {
          if (myid == 0) {
              start_time = MPI_Wtime();
          }
          auto c = PoissonMultipoleOperator(MPI_COMM_WORLD, dfes.get(), &fes, lmax, dom_marker);
          if (myid == 0) {
              end_time = MPI_Wtime();
              assembly_time = (end_time - start_time);
          }
          auto rhof = GridFunction(dfes.get());
          rhof.ProjectCoefficient(rhs_coeff); ///
          c.AddMult(rhof, b, -1);
      } else {
          if (myid == 0) {
              start_time = MPI_Wtime();
          }
          auto c = PoissonLinearisedMultipoleOperator(MPI_COMM_WORLD, vfes.get(), &fes, lmax);
          if (myid == 0) {
              end_time = MPI_Wtime();
              assembly_time = (end_time - start_time);
          }
          const real_t scale_lin = 1.0 * G; ///
          c.AddMult(*u, b, -scale_lin);
      }
  }

  x = 0.0;
  Array<int> ess_tdof_list{};
  if (method == 10) fes.GetEssentialTrueDofs(boundary_marker, ess_tdof_list);

  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  HypreParMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = HypreBoomerAMG(As);

  auto solver = CGSolver(MPI_COMM_WORLD);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(5000);
  solver.SetPrintLevel(1);

  if (method == 1) {
    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    auto c = PoissonDtNOperator(MPI_COMM_WORLD, &fes, lmax);
    if (myid == 0) {
      end_time = MPI_Wtime();
      assembly_time = (end_time - start_time);
    }
    auto C = c.RAP();
    auto D = SumOperator(&A, 1.0, &C, 1.0, false, false);
    solver.SetOperator(D);
    solver.SetPreconditioner(P);

    if (myid == 0) {
      start_time = MPI_Wtime();
    }
    if (dim == 2) {
      auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
      orthoSolver.SetSolver(solver);
      orthoSolver.Mult(B, X);
    } else {
        solver.Mult(B, X);
    }
    if (myid == 0) {
        end_time = MPI_Wtime();
        solver_time = (end_time - start_time);
    }

  } else {
      if (myid == 0) {
          start_time = MPI_Wtime();
      }
      solver.SetOperator(A);
      solver.SetPreconditioner(P);
      if (method == 10) {
          solver.Mult(B, X);
      } else {
          auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
          orthoSolver.SetSolver(solver);
          orthoSolver.Mult(B, X);
      }
      if (myid == 0) {
          end_time = MPI_Wtime();
          solver_time = (end_time - start_time);
      }
  }

  if (myid == 0) {
      std::cout << "Assembly time: " << assembly_time << " s" << std::endl;
      std::cout << "Solver time: " << solver_time << " s" << std::endl;
  }

  a.RecoverFEMSolution(X, b, x);

  //nd.UnscaleGravityPotential(x);

  {
      ostringstream mesh_name, sol_name;
      mesh_name << "results/mesh_prem." << setfill('0') << setw(6) << myid;
      sol_name << "results/sol_prem." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
  }

  {
    const int n = 500;
    const double r0 = 0.0, r1 = 1.2;
    int point_ordering = Ordering::byNODES;

    Vector vxyz;
    vxyz.SetSize(n * dim);
    for (int i = 0; i < n; ++i)
    {
        double r = r0 + (r1 - r0) * i / (n - 1);
        vxyz[i] = r;
        if (dim > 1) vxyz[i + n] = 0.0;
        if (dim > 2) vxyz[i + 2*n] = 0.0;
    }

    FindPointsGSLIB finder(MPI_COMM_WORLD);
    finder.Setup(pmesh);
    finder.SetDistanceToleranceForPointsFoundOnBoundary(10);
    finder.FindPoints(vxyz, point_ordering);

    Vector interp;
    finder.Interpolate(x, interp);
    if (interp.UseDevice()) interp.HostReadWrite();
    vxyz.HostReadWrite();

    if (myid == 0)
    {
        std::ofstream ofs("results/prem_phi_vs_r.dat");
        ofs.setf(std::ios::scientific);
        ofs.precision(12);
        for (int i = 0; i < n; ++i)
        {
            double r_nd = r0 + (r1 - r0) * i / (n - 1);
            ofs << r_nd << " " << interp[i] << "\n";
        }
        ofs.close();
    }
}
  

}

