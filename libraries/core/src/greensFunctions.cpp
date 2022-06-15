#include "greensFunctions.h"

namespace fwi
{
    namespace core
    {
        const double pi = std::atan(1.0) * 4.0;

        namespace greensFunctions
        {
            double Helmholtz2DReal(double k, double r)
            {
                double value = 0.0;
                if(r != 0.0)
                {
                    value = -0.25 * std::cyl_neumann(0, k * r) * k *
                            k;   // Babak 2018 10 25: Equation ID "GreensFunc2d": (first part of rhs). y0 indicates the Bessel function of the first kind.
                }
                return value;
            }

            double Helmholtz2DImag(double k, double r)
            {
                double value = 0.0;
                if(r != 0.0)
                {
                    value = 0.25 * std::cyl_bessel_j(0, k * r) * k *
                            k;   // Babak 2018 10 25: Equation ID "GreensFunc2d": (second part of rhs). j0 indicates the Bessel function of the second kind.
                }
                return value;
            }

            std::complex<double> Helmholtz2D(double k, double r)
            {
                return std::complex<double>(
                    Helmholtz2DReal(k, r), Helmholtz2DImag(k, r));   // Babak 2018 10 25: Equation ID "GreensFunc2d": Creating a complex Greens function.
            }

            double Helmholtz3DReal(double k, double r)
            {
                double value = 0.0;
                if(r != 0.0)
                {
                    value = 0.25 / (r * pi) * std::cos(k * r) * k *
                            k;   // Babak 2018 10 25: Equation ID "GreensFunc2d": real pardouble of Greens function from a 3D Helmhotz equation.
                }
                return value;
            }

            double Helmholtz3DImag(double k, double r)
            {
                double value = 0.0;
                if(r != 0.0)
                {
                    value = 0.25 / (r * pi) * std::sin(k * r) * k *
                            k;   // Babak 2018 10 25: Equation ID "GreensFunc2d": imnaginary pardouble of Greens function from a 3D Helmhotz equation
                }
                return value;
            }

            std::complex<double> Helmholtz3D(double k, double r)
            {
                return std::complex<double>(
                    Helmholtz3DReal(k, r), Helmholtz3DImag(k, r));   // Babak 2018 10 25: Equation ID "GreensFunc2d": Creating complex Greens fucntion
            }

        }   // namespace greensFunctions
    }       // namespace core
}   // namespace fwi
