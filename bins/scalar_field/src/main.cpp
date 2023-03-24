#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

int main() {
    Triangulation<2> triangulation;
 
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);
 
    std::ofstream out("grid-1.svg");
    GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
    std::cout << "Grid written to grid-1.svg" << std::endl;
}