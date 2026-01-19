#include "Stokes.hpp"

static constexpr unsigned int dim = Stokes::dim;

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto nu = [](const Point<dim> & /*p*/) {return 1.0;}; // Viscosity coefficient
  const auto sigma = [](const Point<dim> & /*p*/) {return 0.0;}; // sigma * u
  const auto f = [](const Point<dim> & /*p*/) {
    Tensor<1, dim> f_val;
    f_val[0] = 0.0;
    f_val[1] = 0.0;
    f_val[2] = 0.0;
    return f_val;
  }; // Forcing term
  const auto h = [](const Point<dim> & /*p*/) {
    Tensor<1, dim> h;
    h[0] = 0.0;
    h[1] = 0.0;
    h[2] = 0.0;
    return h;}; // Boundary condition

  const std::string  mesh_file_name  = "../mesh/mesh-step-5.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  Stokes problem(mesh_file_name, degree_velocity, degree_pressure, nu, sigma, f, h);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}