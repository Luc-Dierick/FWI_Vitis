/*
 * greensFunctions.h
 *
 *  Created on: Jun 25, 2015
 *      Author: Michiel de Reus
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>

namespace fwi
{
    namespace core
    {
        namespace greensFunctions
        {
            float Helmholtz2DReal(float k, float r);

            float Helmholtz2DImag(float k, float r);

            double Helmholtz2DReal(double k, double r);

            double Helmholtz2DImag(double k, double r);

            std::complex<double> Helmholtz2D(double k, double r);

            double Helmholtz3DReal(double k, double r);

            double Helmholtz3DImag(double k, double r);

            std::complex<double> Helmholtz3D(double k, double r);
        }   // namespace greensFunctions
    }       // namespace core
}   // namespace fwi
