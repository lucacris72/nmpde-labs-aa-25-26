#include <iostream>

#include <deal.II/base/convergence_table.h>

#include "Full.hpp"

static constexpr unsigned int dim = Full::dim;

// Exact solution. Both value and gradient are known.
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
    return std::sin(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;

    result[0] =
      2.0 * M_PI * std::cos(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
    result[1] =
      4.0 * M_PI * std::sin(2.0 * M_PI * p[0]) * std::cos(4.0 * M_PI * p[1]);

    return result;
  }

  static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
};

// Main function.

// REMEMBER TO CHANGE: mesh, r, mu, b, sigma, f, h
int
main(int /*argc*/, char * /*argv*/[])
{
  const std::vector<unsigned int> N_el_values = {5, 10, 20, 40};
  const ExactSolution exact_solution;
  ConvergenceTable table;

  const unsigned int r              = 1;

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; }; // Diffusion coefficient
  const auto b  = [](const Point<dim> & p) { 
    Tensor<1, dim> result;
    result[0] = 0.0 * p[0];
    result[1] = 0.0 * p[1];
    return result;
   }; // Advection coefficient
  const auto sigma = [](const Point<dim> & p) { return 0.0 * p[0]; }; // Reaction coefficient
  const auto f  = [](const Point<dim>  &/*p*/) { return -5.0; }; // Forcing term
  const auto h  = [](const Point<dim> &p) { return p[1]; }; // Neumann boundary condition

  for (const auto &N_el : N_el_values)
  {
    const std::string mesh_file_name =
      "../mesh/mesh-square-" + std::to_string(N_el) + ".msh";

    Full problem(mesh_file_name, r, mu, b, sigma, f, h);

    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

    const double h = 1.0 / N_el;

    const double error_L2 =
      problem.compute_error(VectorTools::L2_norm, exact_solution);
    const double error_H1 =
      problem.compute_error(VectorTools::H1_norm, exact_solution);

    table.add_value("h", h);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);
  }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);
  table.write_text(std::cout);

  return 0;
}