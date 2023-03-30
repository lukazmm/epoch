#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <utility>
#include <vector>
#include <functional>

template<int dim>
class Domain {
public:
    // Constructors

    Domain(double radius) 
        : m_tria(), 
            m_fe(2), 
            m_dof_handler(), 
            m_quadrature(m_fe.degree + 1)
    {
        using namespace dealii;

        // GridGenerator::hyper_shell(m_tria, Point<dim>(), 0.001, radius, 10, true);
        // m_tria.refine_global(3);

        GridGenerator::hyper_ball(m_tria, Point<dim>(), radius, true);
        m_tria.refine_global(3);

        // GridGenerator::subdivided_hyper_cube<dim>(m_tria, 1000, 0.00, radius, true);

        m_dof_handler.reinit(m_tria);
        m_dof_handler.distribute_dofs(m_fe);

        DoFRenumbering::Cuthill_McKee(m_dof_handler);

        m_constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
        m_constraints.close();

        DynamicSparsityPattern dsp(m_dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(
            m_dof_handler,
            dsp,
            m_constraints,
            /*keep_constrained_dofs = */ false
        );
 
        m_pattern.copy_from(dsp);
        m_system.reinit(m_pattern);
        m_rhs.reinit(m_dof_handler.n_dofs());
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

    void reset_system() {
        m_system = 0.;
        m_rhs = 0.;
    }

    void cell_to_system(
        const dealii::FullMatrix<double>& cell_system, 
        const std::vector<dealii::types::global_dof_index>& local_dof_indices
    ) {
        this->m_constraints.distribute_local_to_global(cell_system, local_dof_indices, m_system);
    }

    void cell_to_rhs(
        const dealii::Vector<double>& cell_rhs, 
        const std::vector<dealii::types::global_dof_index>& local_dof_indices
    ) {
        this->m_constraints.distribute_local_to_global(cell_rhs, local_dof_indices, m_rhs);
    }

    const dealii::QGauss<dim>& quadrature() const {
        return this->m_quadrature;
    }

    dealii::QGauss<dim>& quadrature() {
        return this->m_quadrature;
    }

    const dealii::SparsityPattern& pattern() const {
        return m_pattern;
    }

    const dealii::SparseMatrix<double>& system() const {
        return m_system;
    }

    void solve(dealii::Vector<double>& x) {
        x = 0.0;
        // m_preconditioner.initialize(m_system);
        // m_solver.solve(m_system, x, m_rhs, m_preconditioner);
        // m_constraints.distribute(x);

        m_solver_direct.initialize(m_system);
        m_solver_direct.vmult(x, m_rhs);
        m_constraints.distribute(x);

        // dealii::SolverControl control(1000, m_rhs.l2_norm() * 10e-10);
        // dealii::SolverGMRES<dealii::Vector<double>> solver(control);
        // dealii::PreconditionJacobi<dealii::SparseMatrix<double>> preconditioner;

        // preconditioner.initialize(m_system, 1.0);
        // solver.solve(m_system, x, m_rhs, preconditioner);
    }
    
private:
    dealii::Triangulation<dim> m_tria;

    dealii::FE_Q<dim> m_fe;
    dealii::DoFHandler<dim> m_dof_handler;

    dealii::AffineConstraints<double> m_constraints;

    dealii::QGauss<dim> m_quadrature;

    dealii::SparsityPattern m_pattern;
    dealii::SparseMatrix<double> m_system;
    dealii::Vector<double> m_rhs;

    // dealii::SolverControl m_solver_control;
    // // dealii::PreconditionSSOR<dealii::SparseMatrix<double>> m_preconditioner;
    // dealii::SparseILU<double> m_preconditioner;
    // // dealii::PreconditionJacobi<dealii::SparseMatrix<double>> m_preconditioner;
    // dealii::SolverGMRES<dealii::Vector<double>> m_solver;
    dealii::SparseDirectUMFPACK m_solver_direct;

    
};