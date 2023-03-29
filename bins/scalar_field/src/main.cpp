#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include "domain.hpp"
#include "field.hpp"
#include "metric.hpp"

using namespace dealii;

int main() {
    Domain<2> domain(5.0);

    std::ofstream grid_file("grid.svg");
    GridOut grid_out;
    grid_out.write_svg(domain.tria(), grid_file);


    Field<2> field = Field<2>::gaussian(domain, 0.0, 1.0, 1.0);
    Metric<2> metric = Metric<2>::empty(domain);
    metric.constraints(domain, field);

    std::cout << domain.n_dofs() << std::endl;

    {
        DataOut<2> data_out;
        data_out.attach_dof_handler(domain.dofs());
        data_out.add_data_vector(metric.gamma(), "gamma");
        // data_out.add_data_vector(field.psi(), "psi");
        // data_out.add_data_vector(field.pi(), "pi");

        data_out.build_patches();

        std::ofstream gamma_file("gamma.eps");
        data_out.write_eps(gamma_file);
    }

     {
        DataOut<2> data_out;
        data_out.attach_dof_handler(domain.dofs());
        data_out.add_data_vector(metric.lapse(), "lapse");
        // data_out.add_data_vector(field.psi(), "psi");
        // data_out.add_data_vector(field.pi(), "pi");
    
        data_out.build_patches();
    
        std::ofstream lapse_file("lapse.eps");
        data_out.write_eps(lapse_file);
    }

    
}


// int main() {
//     Domain<1> domain(5.0);

//     // std::ofstream pattern_file("pattern.svg");
//     // domain.pattern().print_svg(pattern_file);

//     Field<1> field = Field<1>::gaussian(domain, 0.0, 1.0, 1.0);
//     Metric<1> metric = Metric<1>::empty(domain);
//     metric.constraints(domain, field);

//     std::cout << domain.n_dofs() << std::endl;

//     DataOut<1> data_out;
//     data_out.attach_dof_handler(domain.dofs());
//     data_out.add_data_vector(metric.gamma(), "gamma");
//     // data_out.add_data_vector(field.psi(), "psi");
//     // data_out.add_data_vector(field.pi(), "pi");

//     data_out.build_patches();

//     std::ofstream gamma_file("gamma.gnuplot");
//     data_out.write_gnuplot(gamma_file);
// }