#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include "domain.hpp"
#include "field.hpp"

using namespace dealii;

int main() {
    Domain<2> domain(0.001, 5.0, 1000);
    Field<2> field = Field<2>::gaussian(domain, 0.0, 1.0, 1.0);

    std::cout << domain.n_dofs() << std::endl;

    DataOut<2> data_out;
    data_out.attach_dof_handler(domain.dofs());
    data_out.add_data_vector(field.psi(), "psi");
    // data_out.add_data_vector(field.psi(), "psi");
    // data_out.add_data_vector(field.pi(), "pi");

    data_out.build_patches();

    std::ofstream output("psi.eps");
    data_out.write_eps(output);
}