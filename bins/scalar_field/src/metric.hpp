#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "common.hpp"
#include "domain.hpp"
#include "field.hpp"

template<int dim>
class Metric {
public:
    Metric() 
    {
    }

    static Metric<dim> empty(const Domain<dim>& domain) {
        Metric<dim> metric;
        metric.m_gamma.reinit(domain.n_dofs());
        metric.m_lapse.reinit(domain.n_dofs());
        metric.m_delta.reinit(domain.n_dofs());

        return metric;
    }

    void constraints(Domain<dim>& domain, const Field<dim>& field) {
        // Solve for Gamma and Lapse

        // Include the dealii namespace
        using namespace dealii;

        // Max number of iterations to converge to gamma (which is nonlinear)
        const unsigned int MAX_ITERATIONS = 10;
        const double CONV_TOLERANCE = 10e-10;

        // FE objects
        FEValues<dim> fe_values(domain.fe(), domain.quadrature(), 
                                        update_values | 
                                        update_quadrature_points | 
                                        update_gradients | 
                                        update_JxW_values);

        const unsigned int dofs_per_cell = mesh.fe().n_dofs_per_cell(); 

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_system(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        // Sratch buffers used to store values and gradients of functions
        std::vector<double> phi_values(dofs_per_cell);
        std::vector<double> psi_values(dofs_per_cell);
        std::vector<double> pi_values(dofs_per_cell);

        std::vector<double> gamma_values(dofs_per_cell);
        std::vector<Tensor<1, dim, double>> gamma_gradients(dofs_per_cell);

        // Solve for Gamma

        // Initial Guess
        VectorTools::project<dim>(
            domain.dofs(),
            domain.constraints(),
            domain.quadrature(),
            ConstantFunction<dim>(1.0),
            m_gamma
        );

        // Refine
        for (unsigned int i = 0; i < MAX_ITERATIONS;i++) {
            m_delta = 0;

            for (const auto& cell : domain.dofs().active_cell_iterators())
            {
                cell_system = 0.;
                cell_rhs = 0.;

                fe_values.reinit(cell);

                fe_values->get_function_values(field.phi(), phi_values);
                fe_values->get_function_values(field.psi(), psi_values);
                fe_values->get_function_values(field.pi(), pi_values);

                fe_values->get_function_values(m_gamma, gamma_values);
                fe_values->get_function_gradients(m_gamma, gamma_gradients);

                const std::function<double(unsigned int)> sys_rad =
                    [](unsigned int q_index) -> double {
                        return 2.0;
                    };

                cell_system_radial(fe_values, sys_rad, cell_system);

                const std::function<double(unsigned int)> sys_val =
                    [&](unsigned int q_index) -> double {
                        const Point<dim>& point = fe_values.quadrature_point(q_index);
                        const double r = point.norm();

                        if r < std::numeric_limits<double>::epsilon() {
                            return 1.0;
                        }

                        const double phi_val = phi_values[q_index];
                        const double psi_val = psi_values[q_index];
                        const double pi_val = pi_values[q_index];

                        const double gamma_val = gamma_values[q_index];

                        const double k = field.kinetic(psi_val, pi_val);
                        const double v = field.potential(phi_val);

                        const double a = r * r * v - 1.0;
                        const double b = r * r * k + 1.0;

                        return -(b + 3.0 * gamma_val * gamma_val * a);
                    };
                
                cell_system_value(fe_values, sys_val, cell_system);

                const std::function<double(unsigned int)> rhs_val =
                    [&](unsigned int q_index) -> double {
                        const Point<dim>& point = fe_values.quadrature_point(q_index);
                        const double r = point.norm();

                        if r < std::numeric_limits<double>::epsilon() {
                            return 0.0;
                        }

                        const double phi_val = phi_values[q_index];
                        const double psi_val = psi_values[q_index];
                        const double pi_val = pi_values[q_index];

                        const double gamma_val = gamma_values[q_index];
                        const double gamma_grad = gamma_gradients[q_index];

                        const double gamma_radial = scalar_product(gamma_grad, point);

                        const double k = field.kinetic(psi_val, pi_val);
                        const double v = field.potential(phi_val);

                        const double a = r * r * v - 1.0;
                        const double b = r * r * k + 1.0;

                        return a * gamma_val * gamma_val * gamma_val + b * gamma_val - 2.0 * gamma_radial;
                    };

                cell_rhs_value(fe_values, rhs_val, cell_rhs);

                cell->get_dof_indices(local_dof_indices);

                domain.cell_to_system(cell_system, local_dof_indices);
                domain.cell_to_rhs(cell_rhs, local_dof_indices);
            }

            domain.solve(m_delta);

            m_gamma += m_delta;

            if m_delta.norm_sq() < CONV_TOLERANCE {
                break;
            }

            if i == MAX_ITERATIONS - 1 {
                std::cout << "Gamma Failed to Converge!" << std::endl;
            }
        }

        // Solve for Lapse
        for (const auto& cell : domain.dofs().active_cell_iterators())
        {
            cell_system = 0.;
            cell_rhs = 0.;

            fe_values.reinit(cell);

            fe_values->get_function_values(field.phi(), phi_values);
            fe_values->get_function_values(field.psi(), psi_values);
            fe_values->get_function_values(field.pi(), pi_values);
            fe_values->get_function_values(m_gamma, gamma_values);

            const std::function<double(unsigned int)> sys_rad =
                [](unsigned int q_index) -> double {
                    return 2.0;
                };

            cell_system_radial(fe_values, sys_rad, cell_system);

            const std::function<double(unsigned int)> sys_val =
                [&](unsigned int q_index) -> double {
                    const Point<dim>& point = fe_values.quadrature_point(q_index);
                    const double r = point.norm();

                    if r < std::numeric_limits<double>::epsilon() {
                        return 0.0;
                    }

                    const double phi_val = phi_values[q_index];
                    const double psi_val = psi_values[q_index];
                    const double pi_val = pi_values[q_index];

                    const double gamma_val = gamma_values[q_index];

                    const double k = field.kinetic(psi_val, pi_val);
                    const double v = field.potential(phi_val);

                    const double a = r * r * v - 1.0;
                    const double c = r * r * k - 1.0;

                    return (gamma_val * gamma_val * a - c);
                };
            
            cell_system_value(fe_values, sys_val, cell_system);

            const std::function<double(unsigned int)> rhs_val =
                [&](unsigned int q_index) -> double {
                    const Point<dim>& point = fe_values.quadrature_point(q_index);
                    const double r = point.norm();

                    if r < std::numeric_limits<double>::epsilon() {
                        return 1.0;
                    }

                    return 0.0;
                };

            cell_rhs_value(fe_values, rhs_val, cell_rhs);

            cell->get_dof_indices(local_dof_indices);

            domain.cell_to_system(cell_system, local_dof_indices);
            domain.cell_to_rhs(cell_rhs, local_dof_indices);
        }

        domain.solve(m_lapse);
    }
    
private:
    dealii::Vector<double> m_gamma;
    dealii::Vector<double> m_lapse;
    dealii::Vector<double> m_delta;
};