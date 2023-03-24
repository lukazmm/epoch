#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>

#include <utility>

template<int dim>
class Domain {
public:
    // Constructors

    Domain(double inner_radius, double outer_radius, unsigned int n) 
        : m_tria(), m_fe(2), m_dof_handler()
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
    
private:
    dealii::Triangulation<dim> m_tria;

    dealii::FE_Q<dim> m_fe;
    dealii::DoFHandler<dim> m_dof_handler;

    dealii::AffineConstraints<double> m_constraints;
};