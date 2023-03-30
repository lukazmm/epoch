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
        metric.m_factor.reinit(domain.n_dofs());

        return metric;
    }

    void constraints(Domain<dim>& domain, const Field<dim>& field) {
        // Solve for Gamma and Lapse

        // Include the dealii namespace
        using namespace dealii;

        // Max number of iterations to converge to gamma (which is nonlinear)
        const unsigned int MAX_ITERATIONS = 20;
        const double CONV_TOLERANCE = 10e-10;

        // FE objects
        FEValues<dim> fe_values(domain.fe(), domain.quadrature(), 
                                        update_values | 
                                        update_quadrature_points | 
                                        update_gradients | 
                                        update_JxW_values);

        const unsigned int dofs_per_cell = domain.fe().n_dofs_per_cell(); 

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_system(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        // Sratch buffers used to store values and gradients of functions
        std::vector<double> phi_values(dofs_per_cell);
        std::vector<double> psi_values(dofs_per_cell);
        std::vector<double> pi_values(dofs_per_cell);

        std::vector<double> values_1(dofs_per_cell);
        std::vector<double> values_2(dofs_per_cell);
        std::vector<Tensor<1, dim, double>> gradients_1(dofs_per_cell);

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
            domain.reset_system();

            m_delta = 0;

            std::cout << "Gamma Iteration: " << i << std::endl;

            for (const auto& cell : domain.dofs().active_cell_iterators())
            {

                fe_values.reinit(cell);

                fe_values.get_function_values(field.phi(), phi_values);
                fe_values.get_function_values(field.psi(), psi_values);
                fe_values.get_function_values(field.pi(), pi_values);

                fe_values.get_function_values(m_gamma, values_1);
                fe_values.get_function_gradients(m_gamma, gradients_1);

                const std::function<void(unsigned int, double&, double&, double&)> sys = 
                    [&](unsigned int q_index, double& grad, double& val, double& rhs) {
                        const Point<dim>& point = fe_values.quadrature_point(q_index);
                        const double r = point.norm();

                        const double phi_val = phi_values[q_index];
                        const double psi_val = psi_values[q_index];
                        const double pi_val = pi_values[q_index];

                        const double gamma_val = values_1[q_index];
                        const Tensor<1, dim, double>& gamma_grad = gradients_1[q_index];
                        const double gamma_radial = scalar_product(gamma_grad, point);

                        const double p = field.potential(phi_val);
                        const double k = field.kinetic(psi_val, pi_val);

                        const double f = 3.0 * gamma_val * gamma_val * (1.0 - r*r*p) - (1.0 + r*r*k);
                        const double g = gamma_val * gamma_val * gamma_val * (r*r*p - 1.0) + gamma_val * (r*r*k + 1.0) - 2.0 * gamma_radial;

                        grad = 2.0;
                        val = f;
                        rhs = g;
                    };

                cell_radial_and_value(fe_values, sys, cell_system, cell_rhs);

                cell->get_dof_indices(local_dof_indices);

                domain.cell_to_system(cell_system, local_dof_indices);
                domain.cell_to_rhs(cell_rhs, local_dof_indices);
            }

            domain.solve(m_delta);

            m_gamma += m_delta;

            std::cout << "Residual is: " << m_delta.norm_sqr() << std::endl;

            if (m_delta.norm_sqr() < CONV_TOLERANCE) {
                break;
            }

            if (i == MAX_ITERATIONS - 1) {
                std::cout << "Gamma Failed to Converge!" << std::endl;
            }
        }

        std::cout << "Solving for Lapse" << std::endl;

        // Solve for Lapse

        domain.reset_system();

        for (const auto& cell : domain.dofs().active_cell_iterators())
        {
            fe_values.reinit(cell);

            fe_values.get_function_values(field.phi(), phi_values);
            fe_values.get_function_values(field.psi(), psi_values);
            fe_values.get_function_values(field.pi(), pi_values);

            fe_values.get_function_values(m_gamma, values_1);

            const std::function<double(unsigned int)> rad =
                [&](unsigned int q_index) -> double {
                    const Point<dim>& point = fe_values.quadrature_point(q_index);
                    const double r = point.norm();

                    const double phi_val = phi_values[q_index];
                    const double psi_val = psi_values[q_index];
                    const double pi_val = pi_values[q_index];

                    const double gamma_val = values_1[q_index];

                    const double v = field.potential(phi_val);
                    const double k = field.kinetic(psi_val, pi_val);

                    const double a = r * r * v - 1.0;
                    const double c = r * r * k - 1.0;

                    return (c - gamma_val * gamma_val * a) / (2.0);
                };

            cell_radial(fe_values, rad, cell_system, cell_rhs);

            cell->get_dof_indices(local_dof_indices);

            domain.cell_to_system(cell_system, local_dof_indices);
            domain.cell_to_rhs(cell_rhs, local_dof_indices);
        }

        domain.solve(m_lapse);

        domain.reset_system();

        for (const auto& cell : domain.dofs().active_cell_iterators())
        {
            fe_values.reinit(cell);

            fe_values.get_function_values(m_lapse, values_1);

            const std::function<double(unsigned int)> val =
                [&](unsigned int q_index) -> double {
                    const double lapse_val = values_1[q_index];

                    return std::exp(lapse_val);
                };

            cell_value(fe_values, val, cell_system, cell_rhs);

            cell->get_dof_indices(local_dof_indices);

            domain.cell_to_system(cell_system, local_dof_indices);
            domain.cell_to_rhs(cell_rhs, local_dof_indices);
        }

        domain.solve(m_lapse);
    }

    const dealii::Vector<double>& gamma() const {
        return m_gamma;
    }

    dealii::Vector<double>& gamma() {
        return m_gamma;
    }

    const dealii::Vector<double>& lapse() const {
        return m_lapse;
    }

    dealii::Vector<double>& lapse() {
        return m_lapse;
    }
    
private:
    dealii::Vector<double> m_gamma;
    dealii::Vector<double> m_lapse;
    dealii::Vector<double> m_delta;
    dealii::Vector<double> m_factor;
};