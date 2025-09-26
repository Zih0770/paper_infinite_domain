/******************************************************************************

Solves the Poisson equation on a whole space through use of a Dirichlet to
Neumann mapping as implemented in the PoissonDtNOperator class.

This is the parallel version of ex2 that makes use of MPI.

The mesh is required to have a spherical exterior boundary. It must have two
attributes. Attirbute 1 is an inner domain with spherically boundary on in which
the density is equal to 1. Attribute 2 is the remainder of the domain, and here
the density is equal to zero. The boundary between attirbutes 1 and 2 is
labelled 1, while the exterior boundary is lablled 2. It is on the exterior
boundary that the DtN mapping acts, this being through the addition of a
bilinear form to the weak form of the equations.

Note that the calculations are done in units for which G = 1.

[-m, --mesh]: The mesh. Either 2D or 3D, but must have the attributes as
              described above. Default is circular_offset.msh in the data
              directory.

[-o, --order]: The polynomial order used in the calculations. Default is 1.

[-sr, --serial_refinement]: The number of times to refine the mesh in serial.
                            The default it 0.

[-pr, --parallel_refinement]: The number of times to refine the mesh in
                              parallel. The default it 0.

[-deg, --degree]: The degree used for the DtN mapping. Default is 4.

[-res, --residual]: If equal to 1, the output is the residual between the
                    numerical solution and an exact one. Default is 0.

*******************************************************************************/

#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"
#include "uniform_sphere.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = atan(1) * 4;

static double SampledValueAtGlobal(const ParGridFunction &gf, const Vector &c0, double r, int n)
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

int main(int argc, char *argv[]) {
    // Initialise MPI and Hypre.
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // Set default options.
    const char *mesh_file = "../data/circular_offset.msh";

    int order = 1;
    int serial_refinement = 0;
    int parallel_refinement = 0;
  int degree = 4;
  int residual = 0;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&serial_refinement, "-sr", "--serial_refinement",
                 "number of serial mesh refinements");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements");
  args.AddOption(&degree, "-deg", "--degree", "Order for Fourier exapansion");
  args.AddOption(&residual, "-res", "--residual",
                 "Output the residual from reference solution");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  // Read in mesh in serial.
  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
  {
    for (int l = 0; l < serial_refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  // Check the mesh has two attributes.
  assert(mesh.attributes.Max() == 2);

  // Form the parallel mesh.
  auto pmesh = ParMesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  {
    for (int l = 0; l < parallel_refinement; l++) {
      pmesh.UniformRefinement();
    }
  }

  assert(pmesh.attributes.Max() == 2);

  // Get centroid and radius for inner domain.
  auto c1 = MeshCentroid(&pmesh, Array<int>{1, 0});
  auto [found1, same1, r1] =
      SphericalBoundaryRadius(&pmesh, Array<int>{1, 0}, c1);
  assert(found1 == 1 && same1 == 1);

  // Get centroid and radius for combined domain.
  auto c2 = MeshCentroid(&pmesh, Array<int>{1, 1});
  auto [found2, same2, r2] =
      SphericalBoundaryRadius(&pmesh, Array<int>{0, 1}, c2);
  assert(found2 == 1 && same2 == 1);

  // Set up the finite element space.
  auto fec = H1_FECollection(order, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // Assemble the binlinear form for Poisson's equation.
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble mass-shifted binlinear form for preconditioning.
  auto eps = ConstantCoefficient(0.001);
  auto as = ParBilinearForm(&fes, &a);
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  // Set up the DtN operator.
  auto c = PoissonDtNOperator(MPI_COMM_WORLD, &fes, degree);

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(1);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  // Set the linear form.
  ParLinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
  b.Assemble();

  // If in 2D, add in necessary boundary form.
  if (dim == 2) {
    auto x = ParGridFunction(&fes);
    x = 1.0;
    auto mass = b(x);
    auto bb = ParLinearForm(&fes);
    auto one = ConstantCoefficient(1);
    auto boundary_marker = Array<int>{0, 1};
    bb.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
    bb.Assemble();
    auto length = bb(x);
    b.Add(-mass / length, bb);
  }

  // Scale the linear form.
  b *= -4 * pi;
  double cc = -12.5664;
  //b *= cc;

  // Set up the linear system
  auto x = ParGridFunction(&fes);
  x = 0.0;
  Array<int> ess_tdof_list{};
  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  auto C = c.RAP();
  auto D = SumOperator(dynamic_cast<Operator *>(&A), 1, &C, 1, false, false);

  // Form the preconditioner, using the mass-matrix to make it
  // positive-definite.
  HypreParMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = HypreBoomerAMG(As);

  // Set up the solver.
  auto solver = CGSolver(MPI_COMM_WORLD);
  solver.SetOperator(D);
  solver.SetPreconditioner(P);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Sovler the linear system.
  if (dim == 2) {
    auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
    orthoSolver.SetSolver(solver);
    orthoSolver.Mult(B, X);
  } else {
    solver.Mult(B, X);
  }
  a.RecoverFEMSolution(X, b, x);

  auto exact = UniformSphereSolution(dim, c1, r1);
  auto exact_coeff = exact.Coefficient();

  auto y = ParGridFunction(&fes);
  y.ProjectCoefficient(exact_coeff);
  double y_fac = cc / (4 * pi);
  y *= y_fac; 

  const double r_eps = 0.02; const int nsmp = 12;
  double phi_c = SampledValueAtGlobal(x, c1, r_eps, nsmp); //
  if (myid == 0) { std::cout<<"phi at centre: "<<phi_c<<std::endl; }                                                  
  //phi -= phi_c;

  double exact_c0 = SampledValueAtGlobal(y, c1, r_eps, nsmp);
  if (myid == 0) { std::cout<<"phi_ex at centre: "<<exact_c0<<std::endl; }
  //exact_gf -= exact_c0;

  // In 2D, remove the mean from the solution.
  if (dim == 2) {
    auto l = ParLinearForm(&fes);
    auto z = ParGridFunction(&fes);
    z = 1.0;
    auto one = ConstantCoefficient(1);
    l.AddDomainIntegrator(new DomainLFIntegrator(one));
    l.Assemble();
    auto area = l(z);
    l /= area;
    auto px = l(x);
    x -= px;
    auto py = l(y);
    y -= py;
  }

  if (residual == 1) {
    x -= y;
  }

  // Write to file.
  ostringstream mesh_name, sol_name;
  mesh_name << "mesh." << setfill('0') << setw(6) << myid;
  sol_name << "sol." << setfill('0') << setw(6) << myid;

  ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(8);
  pmesh.Print(mesh_ofs);

  ofstream sol_ofs(sol_name.str().c_str());
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock << "parallel " << num_procs << " " << myid << "\n";
  sol_sock.precision(8);
  sol_sock << "solution\n" << pmesh << x << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlbc\n" << flush;
  } else {
    sol_sock << "keys RRRilmc\n" << flush;
  }
}
