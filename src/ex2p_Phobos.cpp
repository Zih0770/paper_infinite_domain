#include "mfem.hpp"
#include "mfemElasticity.hpp"

#include "common.hpp"
#include "uniform_sphere.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = std::numbers::pi_v<real_t>;

std::tuple<int, int, mfem::real_t> SBR_ser(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0) {
  using namespace mfem;

  const auto rtol = 1e-3;

  auto dim = mesh->Dimension();
  auto fec = H1_FECollection(1, dim);
  auto fes = FiniteElementSpace(mesh, &fec);

  auto radius = real_t{-1};
  auto x = Vector(dim);
  auto found = 0;
  auto same = 1;
  for (auto i = 0; i < mesh->GetNBE(); i++) {
    //cout<<i<<"-th element: "<<endl;
    if (same == 0) break;
    const auto attr = mesh->GetBdrAttribute(i);

    if (bdr_marker[attr - 1] == 1) {
      found = 1;
      const auto* fe = fes.GetBE(i);
      auto* Trans = fes.GetBdrElementTransformation(i);
      const auto& ir = fe->GetNodes();
      for (auto j = 0; j < ir.GetNPoints(); j++) {
        const auto& ip = ir.IntPoint(j);
        Trans->SetIntPoint(&ip);
        Trans->Transform(ip, x);
        auto d = x.DistanceTo(x0);
        if (radius < 0) {
          radius = d;
        } else {
          same = static_cast<int>(std::abs(radius - d) < rtol * radius);
          if (same == 0) {cout<<radius<<" but "<<d<<endl; break;}
        }
      }
    }
  }
  cout<<"Found: "<<found<<endl;
  cout<<"Same: "<<same<<endl;
  cout<<"Radius: "<<radius<<endl;
  return {found, same, radius};
}

std::tuple<int, int, mfem::real_t> SBR(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0) {
  using namespace mfem;

  const auto rtol = 1e-6;
  auto comm = mesh->GetComm();
  auto rank = mesh->GetMyRank();
  auto size = mesh->GetNRanks();

  auto [local_found, local_same, local_radius] =
      SBR_ser(dynamic_cast<Mesh*>(mesh), bdr_marker, x0);

  cout<<"FFound: "<<local_found<<endl;
  cout<<"SSame: "<<local_same<<endl;
  cout<<"RRadius: "<<local_radius<<endl;

  real_t radius;
  auto found = 0;
  auto same = 1;

  if (rank == 0) {
    auto founds = std::vector<int>(size);
    auto sames = std::vector<int>(size);
    auto radii = std::vector<real_t>(size);

    MPI_Gather(&local_found, 1, MPI_INT, founds.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_same, 1, MPI_INT, sames.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, radii.data(), 1,
               MFEM_MPI_REAL_T, 0, comm);

    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1 && sames[i] == 1) {
        found = 1;
        radius = radii[i];
        break;
      }
    }

    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1 && sames[i] == 1) {
        if (std::abs(radius - radii[i]) > rtol * radius) {
          same = 0;
          break;
        }
      }
    }

  } else {
    MPI_Gather(&local_found, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gather(&local_same, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, nullptr, 0, MFEM_MPI_REAL_T,
               0, comm);
  }

  MPI_Bcast(&found, 1, MPI_INT, 0, comm);
  MPI_Bcast(&same, 1, MPI_INT, 0, comm);
  MPI_Bcast(&radius, 1, MFEM_MPI_REAL_T, 0, comm);

  return {found, same, radius};
}

int main(int argc, char *argv[])
{
  Mpi::Init(argc, argv);
  Hypre::Init();
  const int myid   = Mpi::WorldRank();
  const int nprocs = Mpi::WorldSize();

  const char *mesh_file = "mesh/Phobos.msh";
  int order = 3;
  int lmax  = 8;   
  int method = 10;
  int linearised = 0;

  real_t rho_small_dim = 1860.0;
  real_t G_dim = 6.67430e-11;

  real_t L_scale = 11e3;
  real_t RHO_scale = rho_small_dim;
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
  args.AddOption(&rho_small_dim, "--rho-small", "-rs", "Dimensional rho in small region.");
  args.AddOption(&L_scale, "--L", "-L", "Length scale.");
  args.AddOption(&T_scale, "--T", "-T", "Time scale.");
  args.AddOption(&RHO_scale, "--RHO", "-RHO", "Density scale.");
  args.Parse();
  if (!args.Good()) { 
      if (myid==0) args.PrintUsage(cout); 
      return 1; }
  if (myid==0) args.PrintOptions(cout);

  Nondimensionalisation nd(L_scale, T_scale, RHO_scale);
  const real_t rho_small_nd = nd.ScaleDensity(rho_small_dim);
  const real_t G_nd = G_dim * T_scale * T_scale * RHO_scale;
  real_t G = -4.0 * pi * G_dim * RHO_scale * T_scale * T_scale;
  ConstantCoefficient G_coeff(G);  
  if (myid == 0)
  {
      cout<<"RHS dimensionless number G = "<<G<<endl;
  }

  Mesh mesh(mesh_file, 1, 1, true);
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  const int dim = pmesh.Dimension();

  Array<int> dom_marker(pmesh.attributes.Max()); 
  dom_marker = 0;
  dom_marker[0] = 1;

  auto bdr_marker = Array<int>(pmesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[0] = 1;

  auto _x0 = MeshCentroid(&mesh, 3);
  _x0.Print();
  auto _bdr_marker = ExternalBoundaryMarker(&mesh);
  auto [found, same, radius] = SphericalBoundaryRadius(&mesh, _bdr_marker, _x0);
  cout<<"found_final: "<<found<<endl;
  cout<<"same_final: "<<same<<endl;
  cout<<"radius_final: "<<radius<<endl;

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

  auto rho_coeff1 = ConstantCoefficient(rho_small_nd);
  auto rho_coeff = PWCoefficient();
  rho_coeff.UpdateCoefficient(1, rho_coeff1);
  ProductCoefficient rhs_coeff(G_coeff, rho_coeff);

  auto x = ParGridFunction(&fes);

  ParLinearForm b(&fes);

  auto uv = Vector(dim);
  uv = 1.0;

  std::unique_ptr<ParGridFunction> u; 
  if (linearised == 1) {
      auto uCoeff1 = VectorConstantCoefficient(uv);
      auto uCoeff = PWVectorCoefficient(dim);
      uCoeff.UpdateCoefficient(1, uCoeff1);
      u = std::make_unique<ParGridFunction>(vfes.get());
      u->ProjectCoefficient(uCoeff);
  }

  Array<int> boundary_marker(pmesh.bdr_attributes.Max());
  boundary_marker = 0;
  pmesh.MarkExternalBoundaries(boundary_marker);
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
      d.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rhs_coeff)); ///
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
          rhof.ProjectCoefficient(rhs_coeff); 
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
          const real_t scale_lin = rho_small_nd * G;
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
  solver.SetPrintLevel(3);

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

  //after the solve
  auto submesh = ParSubMesh::CreateFromDomain(pmesh, dom_marker);
  auto subfes = ParFiniteElementSpace(&submesh, &fec);
  auto subx = ParGridFunction(&subfes);
  submesh.Transfer(x, subx);

  {
      ostringstream mesh_name, sol_name;
      mesh_name << "results/mesh_Pho." << setfill('0') << setw(6) << myid;
      sol_name << "results/sol_Pho." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
  }

}

