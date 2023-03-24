#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/numbers.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>

#include "domain.hpp"


template<int dim>
class GaussianPhi : public dealii::Function<dim> {
public:
    GaussianPhi(double amplitude, double sigma) :
        m_amplitude(amplitude),
        m_sigma(sigma)
    {
    }

    virtual double value(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, dealii::ExcIndexRange(component, 0, 1));

        using namespace dealii::numbers;

        double r = p.norm();
        double r2 = r * r;
        double sigma2 = m_sigma * m_sigma;
        double ex = -r2 / sigma2;
        return m_amplitude * std::exp(ex);
    }
private:
    double m_amplitude;
    double m_sigma;
};


template<int dim>
class GaussianPsi : public dealii::Function<dim> {
public:
    GaussianPsi(double amplitude, double sigma) :
        m_amplitude(amplitude),
        m_sigma(sigma)
    {
    }

    virtual double value(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, dealii::ExcIndexRange(component, 0, 1));

        using namespace dealii::numbers;

        double r = p.norm();
        double r2 = r * r;
        double sigma2 = m_sigma * m_sigma;
        double ex = -r2 / sigma2;
        return -2.0 * r / sigma2* m_amplitude * std::exp(ex);
    }
private:
    double m_amplitude;
    double m_sigma;  
};

template<int dim>
class Field {
public:
    Field() :
        m_mass(0)
    {
    }


    Field(double mass, dealii::Vector<double> phi, dealii::Vector<double> psi, dealii::Vector<double> pi) :
        m_mass(mass), 
        m_phi(phi), 
        m_psi(psi), 
        m_pi(pi) 
    {
    }

    static Field<dim> gaussian(const Domain<dim>& domain, double mass, double amplitude, double sigma) 
    {
        using namespace dealii;

        Field<dim> field;
        field.m_mass = mass;
        field.m_phi.reinit(domain.n_dofs());
        field.m_psi.reinit(domain.n_dofs());
        field.m_pi.reinit(domain.n_dofs());

        VectorTools::project<dim>(
            domain.dofs(),
            domain.constraints(),
            QGauss<dim>(domain.fe().degree + 1),
            GaussianPhi<dim>(amplitude, sigma),
            field.m_phi
        );

        VectorTools::project<dim>(
            domain.dofs(),
            domain.constraints(),
            QGauss<dim>(domain.fe().degree + 1),
            GaussianPsi<dim>(amplitude, sigma),
            field.m_psi
        );

        VectorTools::project<dim>(
            domain.dofs(),
            domain.constraints(),
            QGauss<dim>(domain.fe().degree + 1),
            ZeroFunction<dim>(),
            field.m_pi
        );
        
        return field;
    }

    const dealii::Vector<double>& phi() const {
        return this->m_phi;
    }

    dealii::Vector<double>& phi() {
        return this->m_phi;
    }

    const dealii::Vector<double>& psi() const {
        return this->m_phi;
    }

    dealii::Vector<double>& psi() {
        return this->m_psi;
    }

    const dealii::Vector<double>& pi() const {
        return this->m_phi;
    }

    dealii::Vector<double>& pi() {
        return this->m_phi;
    }

    double mass() const {
        return this->m_mass;
    }

private:
    double m_mass;
    dealii::Vector<double> m_phi;
    dealii::Vector<double> m_psi;
    dealii::Vector<double> m_pi;
};