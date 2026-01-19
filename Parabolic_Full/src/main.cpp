#include <iostream>

#include <deal.II/base/convergence_table.h>

#include "Parabolic.hpp"

static constexpr unsigned int dim = Parabolic::dim;

// Exact solution. Both value and gradient are known.
// Use this->get_time() to get the current time.
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    return std::exp(-this->get_time()) * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::sin(M_PI * p[2]);
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;
    double common = std::exp(-this->get_time());

    result[0] = common * M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::sin(M_PI * p[2]);
    result[1] = common * M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]) * std::sin(M_PI * p[2]);
    result[2] = common * M_PI * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::cos(M_PI * p[2]);

    return result;
  }

};

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::vector<unsigned int> N_el_values = {10, 20, 40};
  ExactSolution exact_solution;
  ConvergenceTable table;

  const auto mu = [](const Point<dim> & /*p*/, const double & /*t*/) { return 0.1; };
  const auto b  = [](const Point<dim> & /*p*/, const double & /*t*/) {
    Tensor<1, dim> result;
    result[0] = 1.0;
    result[1] = 1.0;
    result[2] = 1.0;
    return result;
   }; // Advection coefficient
  const auto sigma = [](const Point<dim> & /*p*/, const double & /*t*/) { return 1.0; }; // Reaction coefficient
  const auto f  = [](const Point<dim>  &p, const double  &t) {
    const double u = std::exp(-t) * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::sin(M_PI * p[2]);
    const double u_x = M_PI * std::exp(-t) * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::sin(M_PI * p[2]);
    const double u_y = M_PI * std::exp(-t) * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]) * std::sin(M_PI * p[2]);
    const double u_z = M_PI * std::exp(-t) * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::cos(M_PI * p[2]);

    // f = u_t - div(mu grad u) + b.grad u + sigma u
    // u_t = -u
    // - div(mu grad u) = -0.1 * (-3 * pi^2 * u) = 0.3 * pi^2 * u
    // b.grad u = u_x + u_y + u_z
    // sigma u = u
    
    return -u + 0.3 * M_PI * M_PI * u + (u_x + u_y + u_z) + u;
  };
  const auto h  = [](const Point<dim> &/*p*/, const double & /*t*/) { return 0.0; }; // Neumann boundary condition


  for (const auto &N_el : N_el_values)
  {
    const auto mesh_filename = "../mesh/mesh-cube-" + std::to_string(N_el) + ".msh";
    Parabolic problem(/*mesh_filename = */ mesh_filename,
               /* degree = */ 1,
               /* T = */ 1.0,
               /* theta = */ 1.0,
               /* delta_t = */ 0.025,
               mu,
               b,
               sigma,
               h,
               f);

    problem.run();

    exact_solution.set_time(problem.get_time()); // Ensure exact solution time matches
    const double error_L2 =
      problem.compute_error(VectorTools::L2_norm, exact_solution);
    const double error_H1 =
      problem.compute_error(VectorTools::H1_norm, exact_solution);

    table.add_value("h", 1.0 / N_el);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);
  }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);
  table.write_text(std::cout);

  return 0;
}