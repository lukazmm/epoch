#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>

#include <utility>
#include <vector>
#include <functional>

template<int dim>
class Domain {
public:
    // Constructors

    Domain(double inner_radius, double outer_radius, unsigned int n) 
        : m_tria(), 
            m_fe(2), 
            m_dof_handler(), 
            m_quadrature(m_fe.degree + 1),
            m_solver_control(100, 10e-10),
            m_solver(m_solver_control)
    {
        using namespace dealii;

        // GridGenerator::hyper_shell(m_tria, Point<dim>(), inner_radius, outer_radius, n, true);

        GridGenerator::hyper_ball(m_tria, Point<dim>(), outer_radius, true);
        m_tria.refine_global(4);

        // GridGenerator::subdivided_hyper_cube<dim>(m_tria, n, inner_radius, outer_radius, true);

        m_dof_handler.reinit(m_tria);
        m_dof_handler.distribute_dofs(m_fe);

        m_constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
        m_constraints.close();
    }

    const dealii::Triangulation<dim>& tria() const {
        return this->m_tria;
    }

    dealii::Triangulation<dim>& tria() {
        return this->m_tria;
    }

    dealii::types::global_dof_index n_dofs() const {
        return this->m_dof_handler.n_dofs();
    }

    const dealii::FE_Q<dim>& fe() const {
        return this->m_fe;
    }

    dealii::FE_Q<dim>& fe() {
        return this->m_fe;
    }

    const dealii::DoFHandler<dim>& dofs() const {
        return this->m_dof_handler;
    }

    dealii::DoFHandler<dim>& dofs() {
        return this->m_dof_handler;
    }

    const dealii::AffineConstraints<double>& constraints() const {
        return this->m_constraints;
    }

    dealii::AffineConstraints<double>& constraints() {
        return this->m_constraints;
    }

    void cell_to_system(
        const dealii::FullMatrix<double>& cell_system, 
        const std::vector<dealii::types::global_dof_index>& local_dof_indices
    ) {
        this->m_constraints.distribute_local_to_global(cell_system, local_dof_indices, this->m_system);
    }

    void cell_to_rhs(
        const dealii::Vector<double>& cell_rhs, 
        const std::vector<dealii::types::global_dof_index>& local_dof_indices
    ) {
        this->m_constraints.distribute_local_to_global(cell_system, local_dof_indices, this->m_rhs);
    }

    const dealii::QGauss<dim>& quadrature() const {
        return this->m_quadrature;
    }

    dealii::QGauss<dim>& quadrature() {
        return this->m_quadrature;
    }

    void solve(dealii::Vector<double>& x) {
        x = 0.0;
        m_preconditioner.initialize(m_system, 1.2);
        m_solver.solve(m_system, x, m_rhs, m_preconditioner);
        m_constraints.distribute(x);
    }
    
private:
    dealii::Triangulation<dim> m_tria;

    dealii::FE_Q<dim> m_fe;
    dealii::DoFHandler<dim> m_dof_handler;

    dealii::AffineConstraints<double> m_constraints;

    dealii::QGauss<dim> m_quadrature;

    dealii::SparseMatrix<double> m_system;
    dealii::Vector<double> m_rhs;

    dealii::SolverControl<double> m_solver_control;
    dealii::PreconditionSSOR<dealii::SparseMatrix<double>> m_preconditioner;
    dealii::SolverBicgstab<dealii::Vector<double>> m_solver;
};