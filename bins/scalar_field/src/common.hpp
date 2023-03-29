#pragma once

#include <deal.II/base/point.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <utility>
#include <vector>
#include <functional>

template<int dim>
void cell_radial(
    const dealii::FEValues<dim>& fe_values,
    const std::function<double(unsigned)> func,
    dealii::FullMatrix<double>& cell_system,
    dealii::Vector<double>& cell_rhs
) {
    using namespace dealii;

    cell_system = 0.;
    cell_rhs = 0.;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
        const double value = func(q_index);
        const double JxW = fe_values.JxW(q_index);
        const Point<dim>& position = fe_values.quadrature_point(q_index);

        for (const unsigned int i : fe_values.dof_indices()) {
            const Tensor<1, dim>& i_grad = fe_values.shape_grad(i, q_index);
            const double i_radial = scalar_product(i_grad, position);

            for (const unsigned int j : fe_values.dof_indices()) {
                const Tensor<1, dim>& j_grad = fe_values.shape_grad(j, q_index);
                const double j_radial = scalar_product(j_grad, position);

                cell_system(i, j) +=  i_radial * j_radial * JxW;
            }

            cell_rhs(i) += value * i_radial * JxW;
        }
    }
}


template<int dim>
void cell_value(
    const dealii::FEValues<dim>& fe_values,
    const std::function<double(unsigned)> func,
    dealii::FullMatrix<double>& cell_system,
    dealii::Vector<double>& cell_rhs
) {
    using namespace dealii;

    cell_system = 0.;
    cell_rhs = 0.;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
        const double value = func(q_index);
        const double JxW = fe_values.JxW(q_index);
        const Point<dim>& position = fe_values.quadrature_point(q_index);

        for (const unsigned int i : fe_values.dof_indices()) {
            const double i_val = fe_values.shape_val(i, q_index);

            for (const unsigned int j : fe_values.dof_indices()) {
                const double j_val = fe_values.shape_val(j, q_index);

                cell_system(i, j) +=  i_val * j_val * JxW;
            }

            cell_rhs(i) += value * i_val * JxW;
        }
    }
}

template<int dim>
void cell_system_radial(
    const dealii::FEValues<dim>& fe_values, 
    const std::function<double(unsigned int)> func, 
    dealii::FullMatrix<double>& cell_system
) {
    using namespace dealii;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
        const double value = func(q_index);
        const double JxW = fe_values.JxW(q_index);
        const Point<dim>& position = fe_values.quadrature_point(q_index);

        for (const unsigned int i : fe_values.dof_indices()) {
            const double i_val = fe_values.shape_value(i, q_index);

            for (const unsigned int j : fe_values.dof_indices()) {
                const Tensor<1, dim>& j_grad = fe_values.shape_grad(j, q_index);
                const double j_radial = scalar_product(j_grad, position);
                
                cell_system(i, j) += value * i_val * j_radial * JxW;
            }
        }
    }
}

template<int dim>
void cell_system_value(
    const dealii::FEValues<dim>& fe_values, 
    const std::function<double(unsigned int)> func, 
    dealii::FullMatrix<double>& cell_system
) {
    using namespace dealii;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
        const double value = func(q_index);
        const double JxW = fe_values.JxW(q_index);

        for (const unsigned int i : fe_values.dof_indices()) {
             const double i_val = fe_values.shape_value(i, q_index);

            for (const unsigned int j : fe_values.dof_indices()) {
                const double j_val = fe_values.shape_value(j, q_index);

                cell_system(i, j) += value * i_val * j_val * JxW;
            }
        }
    }
}


template<int dim>
void cell_rhs_value(
    const dealii::FEValues<dim>& fe_values, 
    const std::function<double(unsigned int)> func, 
    dealii::Vector<double>& cell_rhs
) {
    using namespace dealii;

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
        const double value = func(q_index);
        const double JxW = fe_values.JxW(q_index);

        for (const unsigned int i : fe_values.dof_indices()) {
            const double i_val = fe_values.shape_value(i, q_index);

            cell_rhs(i) += value * i_val * JxW;
        }
    }
}