// src/common.hpp
#pragma once

#include "mfem.hpp"
#include <iostream>

using namespace mfem;

class Nondimensionalisation {
private:
    real_t L;   // Length scale [m]
    real_t T;   // Time scale [s]
    real_t RHO; // Density scale [kg/m^3]

public:
    Nondimensionalisation(real_t length_scale, real_t time_scale, real_t density_scale)
        : L(length_scale), T(time_scale), RHO(density_scale) {}

    // Accessors
    real_t Length() const { return L; }
    real_t Time() const { return T; }
    real_t Density() const { return RHO; }

    // Derived scales
    real_t Velocity() const { return L / T; }
    real_t Acceleration() const { return L / (T*T); }
    real_t Pressure() const { return RHO * L*L / (T*T); } // [Pa]
    real_t Gravity() const { return L / (T*T); }
    real_t Potential() const { return L*L / (T*T); }

    // Scaling functions for scalars
    real_t ScaleLength(real_t x) const { return x / L; }
    real_t UnscaleLength(real_t x_nd) const { return x_nd * L; }

    real_t ScaleDensity(real_t rho) const { return rho / RHO; }
    real_t UnscaleDensity(real_t rho_nd) const { return rho_nd * RHO; }

    real_t ScaleGravityPotential(real_t phi) const { return phi / Potential(); }
    real_t UnscaleGravityPotential(real_t phi_nd) const { return phi_nd * Potential(); }

    real_t ScaleStress(real_t sigma) const { return sigma / Pressure(); }
    real_t UnscaleStress(real_t sigma_nd) const { return sigma_nd * Pressure(); }

    // Scaling for GridFunction fields
    void UnscaleGravityPotential(GridFunction &phi_gf) const { phi_gf *= Potential(); }
    void UnscaleDisplacement(GridFunction &u_gf) const { u_gf *= L; }
    void UnscaleStress(GridFunction &sigma_gf) const { sigma_gf *= Pressure(); }

    // Create a scaled density coefficient from a dimensional one
    Coefficient *MakeScaledDensityCoefficient(Coefficient &rho_coeff) const {
        return new ProductCoefficient(1.0 / RHO, rho_coeff);
    }

    void Print() const {
        std::cout << "Scaling parameters:\n";
        std::cout << "  Length scale: " << L << " m\n";
        std::cout << "  Time scale: " << T << " s\n";
        std::cout << "  Density scale: " << RHO << " kg/m^3\n";
        std::cout << "  Gravity potential scale: " << Potential() << " m^2/s^2\n";
    }
};

