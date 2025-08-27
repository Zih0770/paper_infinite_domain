#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid   = Mpi::WorldRank();
   const int nprocs = Mpi::WorldSize();

   const char *mesh_file = "mesh/spherical_offset.msh";
   int order = 2;
   int small_attr  = 0;
   int large_attr  = 1;
   double rho_small = 10.0;
   double rho_large =  1.0;   
   double G = 1.0; // 6.6743e-11;     
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Gmsh mesh file (.msh).");
   args.AddOption(&order, "-o", "--order", "H1 FE order.");
   args.AddOption(&rho_small, "--rho-small", "-rs", "Density in small sphere.");
   args.AddOption(&rho_large, "--rho-large", "-rl", "Density in large sphere.");
   args.AddOption(&small_attr, "--attr-small", "-as", "Region attribute for small sphere (>=1).");
   args.AddOption(&large_attr, "--attr-large", "-al", "Region attribute for large sphere (>=1).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Send solution to GLVis.");
   args.Parse();
   if (!args.Good()) { if (myid==0) args.PrintUsage(cout); return 1; }
   if (myid==0) args.PrintOptions(cout);
   
   Mesh mesh(mesh_file, 1, 1, true);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   //mesh.Clear();

   const int dim = pmesh.Dimension();
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec);
   //if (myid==0) { cout << "Global unknowns: " << fes.GlobalTrueVSize() << endl; }

   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 0;
      pmesh.MarkExternalBoundaries(ess_bdr);
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   const int max_att = pmesh.attributes.Size() ? pmesh.attributes.Max() : 0;
   Vector rho_vals(max_att);
   rho_vals = 0.0;
   rho_vals(small_attr) = rho_small;
   rho_vals(large_attr) = rho_large; 
   PWConstCoefficient rho_pw(rho_vals);

   ConstantCoefficient four_pi_G(4.0 * M_PI * G);
   ProductCoefficient rhs_coeff(four_pi_G, rho_pw);

   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   b.Assemble();

   ParGridFunction phi(&fes); phi = 0.0;

   ConstantCoefficient one(1.0);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B); 

   HypreBoomerAMG amg;
   amg.SetOperator(*A);              
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(myid==0 ? 1 : 0);
   cg.SetPreconditioner(amg);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, phi);

   MPI_Barrier(MPI_COMM_WORLD);
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "results/mesh." << setfill('0') << setw(6) << myid;
      sol_name  << "results/phi."  << setfill('0') << setw(6) << myid;
      ofstream mout(mesh_name.str()); mout.precision(8); pmesh.Print(mout);
      ofstream gout(sol_name.str());  gout.precision(8); phi.Save(gout);
   }

   if (visualization)
   {
      socketstream sol_sock("localhost", 19916);
      sol_sock << "parallel " << nprocs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << phi << flush;
   }

   return 0;
}

